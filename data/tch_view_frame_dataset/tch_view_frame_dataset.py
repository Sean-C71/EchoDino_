import json

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import Config


class TCH_view_frame_Dataset(Dataset):
    def __init__(self,
                 split="train",
                 transform=None,
                 num_picked=None,
                 data_dir=Config.TCH_VIEW_FRAME_DATASET_ROOT,
                 ):

        self.data_dir = data_dir
        self.split = split
        self.transforms = transform

        # get label file
        all_df = pd.read_csv(f"{data_dir}/labels/G4_8_12_viewLabels.csv")
        condition = (all_df['split'] == split)
        self.sample_df = all_df[condition].copy().reset_index()
        with open(f"{data_dir}/labels/combo_mapping.json", 'r', encoding='utf-8') as f:
            self.mapping_dict = json.load(f) # label2idx, idx2label
        self.num_classes=len(self.mapping_dict['label2idx'])
        Config.logger.info(f"{self.num_classes} labels in this dataset.")

        if len(self.sample_df) == 0:
            Config.logger.info(f"No data found for split='{split}'")
            raise ValueError(f"No data found for split='{split}'")

        if num_picked:
            self.sample_df = self.sample_df[:num_picked]

    def __getitem__(self, idx):
        img_row = self.sample_df.iloc[idx]

        # get image
        dcm_path = img_row['path']  # /rdf/transferred_data/original_4400_echos/Group_08/264931/16439430.dcm
        img_path = dcm_path.replace('/rdf/transferred_data/original_4400_echos',
                                    '/rdf/data/RDF/forEchoDino/images').replace('.dcm', '/0.png')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)  # np.array, CHW

        # get label
        group = img_row['group']
        mode_label = img_row['mode_label']
        pos_label = img_row['pos_label']
        view_label = img_row['view_label']
        combo_label = img_row['combo']
        combo_idx= img_row['combo_idx']
        img_name = "_".join(dcm_path.split('/')[-3:])

        # get image
        if self.transforms:
            image = self.transforms(image)

        return {
            "image": image,  # CHW
            "target": combo_idx,
            "mode_label": mode_label,
            "pos_label": pos_label,
            "view_label": view_label,
            "combo_label": combo_label,
            "group": group,
            "img_name": img_name,
        }

    def __len__(self):
        return len(self.sample_df)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = TCH_view_frame_Dataset(
        split="train",
        transform=None,
        num_picked=None,
    )
    print(len(dataset))

    for i in np.random.choice(len(dataset), 10):
        print("*" * 20)
        cur_sample = dataset[i]  # CHW 0-255 int
        print(cur_sample['image'].min(), cur_sample['image'].max())
        print(cur_sample['target'], type(cur_sample['target']))
        print(cur_sample['image'].shape)
        print(f"{cur_sample['mode_label']}|{cur_sample['pos_label']}|{cur_sample['view_label']}")
        print(f"{cur_sample['combo_label']}|{cur_sample['group']}")

        plt.imshow(cur_sample['image'].transpose(1, 2, 0))
        plt.title(f"{cur_sample['mode_label']}|{cur_sample['pos_label']}|{cur_sample['view_label']}")
        plt.show()
