import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
# Accelerator & WandB
from accelerate import Accelerator
from accelerate.utils import set_seed
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import Config
from data import TCH_view_frame_Dataset
from models import Dino_TchViewFrame_classification
from myTrain.tchViewFrame_classification.dino.train import ModelArch, DataTransforms, image_collate_fn


# ---------------------------------------------------------
# Utils & Transforms
# ---------------------------------------------------------
def count_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    Config.logger.info(f"Total: {pytorch_total_params}")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Config.logger.info(f"Trainable: {pytorch_total_params}")


# ---------------------------------------------------------
# Loaders
# ---------------------------------------------------------

def load_model(args):
    # Using args to pass the ckpt_path
    if args.model_type == ModelArch.Dino_TCHVIEW_FRAME_CLS:
        model = Dino_TchViewFrame_classification(ckpt_path=None)
    else:
        msg = f"Wrong model_type: {args.model_type}."
        Config.logger.info(msg)
        raise ValueError(msg)
    Config.logger.info(f"Using {args.model_type} Model")
    Config.logger.info(f"Loading weights from {args.whole_model_path}")

    model.load_state_dict(torch.load(args.whole_model_path, map_location='cpu'), strict=True)
    Config.logger.info(f"Loaded weights successfully")

    return model


def load_dataset(batch_size, args):
    """
    Get dataset and dataloaders
    """
    test_dataset = TCH_view_frame_Dataset(
        split=args.split,
        transform=DataTransforms.get_transform(args.model_type, args.split),
    )
    cur_sample = test_dataset[0]
    Config.logger.info(
        f"image shape: {cur_sample['image'].shape}, label_eg: {cur_sample['target']}")
    Config.logger.info(
        f"{args.split} set size: {len(test_dataset)}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=image_collate_fn,
        num_workers=8,
        pin_memory=True
    )

    return test_loader


# ---------------------------------------------------------
# Inference and Analysis
# ---------------------------------------------------------
def doInfer(model, dataloader, args):
    save_path = os.path.join(
        os.path.dirname(args.whole_model_path),
        f"eval_{args.split}",
        f"eval_{args.split}.csv"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.to('cuda')
    model.eval()

    all_labels = []
    all_preds = []
    all_img_names = []

    # Get mapping from dataset to use in analysis
    # mapping_dict['idx2label'] is { "0": "name", "1": "name" ... }
    idx2label = dataloader.dataset.mapping_dict['idx2label']

    for batch in tqdm(dataloader, desc=f"Evaluating {args.split}"):
        images = batch['images_tensor'].to('cuda')
        labels = batch['targets_tensor']  # This is already shape [B] from our updated collate
        img_names = batch['img_names']

        with torch.no_grad():
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_img_names.extend(img_names)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Save CSV
    df = pd.DataFrame({
        'label': all_labels,
        'prediction': all_preds,
        'img_name': all_img_names,
    })
    df.to_csv(save_path, index=False)
    Config.logger.info(f"Saved prediction to {save_path}")

    # Run Analysis
    doAnalyze(df, idx2label, args)


def doAnalyze(df, idx2label, args):
    eval_dir = os.path.join(os.path.dirname(args.whole_model_path), f"eval_{args.split}")

    # 1. Prepare Label Names
    # JSON keys are strings, convert to int to sort correctly
    sorted_indices = sorted([int(k) for k in idx2label.keys()])
    target_names = [idx2label[str(i)] for i in sorted_indices]

    y_true = df['label'].astype(int)
    y_pred = df['prediction'].astype(int)
    acc = accuracy_score(y_true, y_pred)

    # 2. Generate and Save Classification Report
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4
    )

    header = f"View (Overall, on {args.split} samples) (Accuracy: {acc:.4f}):\n"
    full_report = header + report_text

    report_save_path = os.path.join(eval_dir, f"report_{args.split}.txt")
    with open(report_save_path, "w") as f:
        f.write(full_report)

    Config.logger.info(f"\n{full_report}")

    # 3. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred, labels=sorted_indices)

    # Use log scale for visualization if classes are highly imbalanced
    plt.figure(figsize=(max(12, len(target_names) // 2), max(10, len(target_names) // 2)))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)

    plt.title(f'Confusion Matrix - {args.split} (Acc: {acc:.4f})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    cm_path = os.path.join(eval_dir, f"confusion_matrix_{args.split}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    Config.logger.info(f"Confusion matrix saved to {cm_path}")
    plt.close()


# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='DINO PatchSA Training')
    # Model parameters
    parser.add_argument('--model_type', type=lambda x: ModelArch(x), default="dino_tchViewFrame_cls",
                        help='Specify which model arch to be used')
    parser.add_argument('--whole_model_path', type=str,
                        default="")

    # Data parameters
    parser.add_argument('--split', type=str, default="test", choices=['train', 'val', 'test'])

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)  # Per device batch size

    return parser.parse_args()


def main():
    args = parse_args()
    Config.logger = Config.get_logger(f"{os.path.dirname(args.whole_model_path)}/eval_logs.txt")

    model = load_model(args)
    test_loader = load_dataset(args.batch_size, args)
    doInfer(model, test_loader, args)


if __name__ == '__main__':
    main()
