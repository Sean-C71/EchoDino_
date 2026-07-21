import os
import argparse
from datetime import datetime
from typing import List, Dict, Any
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import v2

# Accelerator & WandB
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
from transformers import get_cosine_schedule_with_warmup

# Our imports
import Config
from models import Dino_TchViewFrame_classification
from data import TCH_view_frame_Dataset


# ---------------------------------------------------------
# Utils & Transforms
# ---------------------------------------------------------
class ModelArch(Enum):
    Dino_TCHVIEW_FRAME_CLS = "dino_tchViewFrame_cls"


class AspectRatioPadResize:
    """
    Resizes image to maintain aspect ratio (longest edge = image_size)
    and pads the shorter edge to create a square.
    """

    def __init__(self, image_size=256):
        self.image_size = image_size

    def __call__(self, x):
        # Expects x shape [..., C, H, W]
        h, w = x.shape[-2:]

        # 1. Calculate new dimensions maintaining aspect ratio
        scale = self.image_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 2. Resize
        x = TF.resize(x, [new_h, new_w], antialias=True)

        # 3. Calculate padding
        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w

        # Padding: (left, top, right, bottom)
        # We split the padding to center the image
        padding = [
            pad_w // 2,
            pad_h // 2,
            pad_w - (pad_w // 2),
            pad_h - (pad_h // 2)
        ]

        # 4. Pad with the minimum value of the current image
        fill_value = x.min().item()
        x = TF.pad(x, padding, fill=fill_value)

        # Final safety check to ensure squareness despite rounding
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = TF.resize(x, [self.image_size, self.image_size])

        return x


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
        model = Dino_TchViewFrame_classification(ckpt_path=args.ckpt_path, train_encoder=args.train_encoder)
    else:
        msg = f"Wrong model_type: {args.model_type}."
        Config.logger.info(msg)
        raise ValueError(msg)
    Config.logger.info(f"Using {args.model_type} Model")

    for name, param in model.named_parameters():
        if 'clsSA_pe' in name:
            param.requires_grad = False  # all require grad first, except the PE of clsSA
        else:
            param.requires_grad = True

    if not args.train_encoder:
        for name, param in model.named_parameters():
            if (
                    "backbone_model" in name
            ):
                param.requires_grad = False
    count_params(model)
    return model


class DataTransforms:
    _transforms = {
        ModelArch.Dino_TCHVIEW_FRAME_CLS: {
            "train": v2.Compose([
                v2.Lambda(lambda x: torch.from_numpy(x)),
                v2.ToDtype(torch.float32, scale=True),
                AspectRatioPadResize(image_size=256),
                v2.RandomRotation((-15, 15)),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]),
            "val": v2.Compose([
                v2.Lambda(lambda x: torch.from_numpy(x)),
                v2.ToDtype(torch.float32, scale=True),
                AspectRatioPadResize(image_size=256),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]),
            "test": v2.Compose([
                v2.Lambda(lambda x: torch.from_numpy(x)),
                v2.ToDtype(torch.float32, scale=True),
                AspectRatioPadResize(image_size=256),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]),
        },
    }

    @classmethod
    def get_transform(cls, model_type: ModelArch, split: str):
        """Return the transform pipeline for the given model and split."""
        Config.logger.info(f"using {model_type} {split} transforms")
        return cls._transforms[model_type][split]


def image_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for processing batches from the dataset.

    Args:
        batch: List of dictionaries from __getitem__.

    Returns:
        Dictionary with:
            - "images_tensor": [B, C, H, W]
            - "targets_tensor": [B, 1]
    """

    # 1. Extract lists using list comprehension (faster/cleaner)
    images_list = [item["image"] for item in batch]
    targets_list = [item["target"] for item in batch]
    img_names = [item["img_name"] for item in batch]

    # 2. Stack Images
    images_tensor = torch.stack(images_list, dim=0)

    # 3. Stack Targets
    targets_tensor = torch.tensor(targets_list, dtype=torch.long)

    return {
        "images_tensor": images_tensor,
        "targets_tensor": targets_tensor,
        "img_names": img_names,
    }


def load_dataset(batch_size, args):
    """
    Get dataset and dataloaders
    """
    trainset = TCH_view_frame_Dataset(
        split="train",
        transform=DataTransforms.get_transform(args.model_type, "train"),
    )
    cur_sample = trainset[0]
    Config.logger.info(
        f"image shape: {cur_sample['image'].shape}, label_eg: {cur_sample['target']}")

    valset = TCH_view_frame_Dataset(
        split="val",
        transform=DataTransforms.get_transform(args.model_type, "val"),
    )
    Config.logger.info(
        f"trainset size: {len(trainset)}, valset size: {len(valset)}"
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=image_collate_fn,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=image_collate_fn,
        num_workers=8,
        pin_memory=True
    )

    return train_loader, val_loader


# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='DINO PatchSA Training')
    # Model parameters
    parser.add_argument('--model_type', type=lambda x: ModelArch(x), default="dino_patchSa",
                        help='Specify which model arch to be used')
    parser.add_argument('--ckpt_path', type=str,
                        default=Config.DINO_DEFAULT_PATH,
                        help="DINO backbone features")
    parser.add_argument('--train_encoder', action='store_true', help='Train the encoder, default not')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)  # Per device batch size
    parser.add_argument('--lr', type=float, default=2e-5)  # Slightly lower LR usually safer for fine-tuning
    parser.add_argument('--weight_decay', type=float, default=0.001)

    # Logging
    parser.add_argument('--wandb_project_name', type=str, default='DINOPatchSA_EF')
    parser.add_argument('--experiment_name', type=str, default='DINOPatchSA_EF_v1')
    parser.add_argument('--output_dir', type=str, required=True, help="Absolute path to save logs and ckpts")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    Config.logger.info(f"Setting output_dir as {args.output_dir}")

    # 1. Setup Accelerator (Enables BF16 and DDP automatically)
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb",
    )
    set_seed(42)

    # 2. Setup Logging (Only on Main Process)
    if accelerator.is_main_process:
        # Create output dir

        # Init Custom Logger
        Config.logger = Config.get_logger(f"{args.output_dir}/logs.txt")
        Config.logger.info(f"Initialized Accelerator. Mixed Precision: {accelerator.mixed_precision}")
        Config.logger.info(args)

        # Init WandB
        accelerator.init_trackers(
            project_name=args.wandb_project_name,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.experiment_name}}
        )

    # 3. Load Data & Model
    train_loader, val_loader = load_dataset(args.batch_size, args)
    model = load_model(args)

    # 4. Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader), eta_min=args.lr*0.001)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_epochs * len(train_loader),
        num_training_steps=args.epochs * len(train_loader),
        num_cycles=0.5
    )
    loss_fn = nn.CrossEntropyLoss()

    # 5. Prepare with Accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_val_acc = 0.0

    if accelerator.is_main_process:
        Config.logger.info("Starting Training...")

    for epoch in range(args.epochs):
        model.train()
        train_loss_accum = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            images = batch['images_tensor']
            labels = batch['targets_tensor']

            # Forward
            logits = model(images)
            loss = loss_fn(logits, labels)

            # Backward
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Scheduler Step
            scheduler.step()

            train_loss_accum += loss.item()

            # CLI Logging (Step-wise)
            if step % 10 == 0 and accelerator.is_main_process:
                norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                Config.logger.info(
                    f"Epoch [{epoch + 1}/{args.epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f} | Grad Norm: {norm_val:.4f}")

                # Log to WandB
                accelerator.log({"train_step_loss": loss.item(), "grad_norm": norm_val},
                                step=epoch * len(train_loader) + step)

        # ---------------------------------------------------------
        # Validation Phase
        # ---------------------------------------------------------
        model.eval()
        val_loss_accum = 0.0
        correct_preds = 0
        total_samples = 0

        for batch in val_loader:
            images = batch['images_tensor']
            labels = batch['targets_tensor']

            with torch.no_grad():
                logits = model(images)
                loss = loss_fn(logits, labels)

            # Stats
            preds = torch.argmax(logits, dim=1)
            # Gather predictions and labels from all GPUs
            all_preds, all_labels, all_loss = accelerator.gather_for_metrics((preds, labels, loss))

            val_loss_accum += all_loss.mean().item()
            correct_preds += (all_preds == all_labels).sum().item()
            total_samples += all_labels.size(0)

        # Calculate epoch metrics
        avg_train_loss = train_loss_accum / len(train_loader)
        avg_val_loss = val_loss_accum / len(val_loader)
        val_acc = correct_preds / total_samples

        # ---------------------------------------------------------
        # Logging & Checkpointing (Main Process Only)
        # ---------------------------------------------------------
        if accelerator.is_main_process:
            # 1. WandB Log
            total_steps_so_far = (epoch + 1) * len(train_loader)
            accelerator.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "epoch": epoch + 1,
                "lr": scheduler.get_last_lr()[0]
            }, step=total_steps_so_far)

            # 2. CLI Log
            Config.logger.info(
                f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # 3. Save Best Model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(args.output_dir, "best_model.pth")

                Config.logger.info(f"New best model found! Saving to {save_path}")

                # Unwrap model to remove DDP wrappers before saving
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), save_path)

    if accelerator.is_main_process:
        Config.logger.info("Training Finished.")
        accelerator.end_training()


if __name__ == '__main__':
    main()
