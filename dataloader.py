import os
import numpy as np
import pathlib
import pandas as pd
import pickle as pkl
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nsml import DATASET_PATH
from sklearn.model_selection import ShuffleSplit


def get_class_weights(add_value=0.0):
    train_label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
    label_matrix = np.load(train_label_path)
    unq_count = np.zeros_like(label_matrix[0])
    y = np.bincount(np.where(label_matrix == 1)[1])
    idx = np.nonzero(y)[0]
    for id, cnt in zip(idx, y[idx]):
        unq_count[id] += cnt
    log_unq_count = np.log10(unq_count)
    weight = 1 - (log_unq_count / np.max(log_unq_count))
    if add_value > 0:
        weight = (weight + add_value) / (1. + add_value)
    return weight


def train_dataloader(input_size=128,
                     batch_size=64,
                     num_workers=0
                     ):
    image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images')
    label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
    train_meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')
    # tags_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'tags.tsv')
    meta_data = pd.read_csv(train_meta_path, delimiter=',', header=0)
    # tags_meta_data = pd.read_csv(tags_path, delimiter='\t', header=0)
    # print(tags_meta_data.iloc[:30])

    label_matrix = np.load(label_path)

    rs = ShuffleSplit(n_splits=1, test_size=.1, random_state=42)
    for train_index, val_index in rs.split(meta_data):
        val_meta_data = meta_data.iloc[val_index]
        train_meta_data = meta_data.iloc[train_index]

        train_labels = label_matrix[train_index]
        val_labels = label_matrix[val_index]

        print("train_x", len(train_meta_data))
        print("val_x", len(val_meta_data))
        print("train_y", len(train_labels))
        print("val_y", len(val_labels))

    train_dataloader = DataLoader(
        AIRushDataset(image_dir, train_meta_data, label_path=train_labels,
                      transform=transforms.Compose(
                          [transforms.Resize((input_size, input_size)), transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    val_dataloader = DataLoader(
        AIRushDataset(image_dir, val_meta_data, label_path=val_labels,
                      transform=transforms.Compose(
                          [transforms.Resize((input_size, input_size)), transforms.ToTensor()])),
        batch_size=batch_size//2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    return train_dataloader, val_dataloader


class AIRushDataset(Dataset):
    def __init__(self, image_data_path, meta_data, label_path=None, transform=None):
        self.meta_data = meta_data
        self.image_dir = image_data_path
        self.label_matrix = label_path
        self.transform = transform

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, str(self.meta_data['package_id'].iloc[idx]),
                                str(self.meta_data['sticker_id'].iloc[idx]) + '.png')
        png = Image.open(img_name).convert('RGBA')
        png.load()  # required for png.split()

        new_img = Image.new("RGB", png.size, (255, 255, 255))
        new_img.paste(png, mask=png.split()[3])  # 3 is the alpha channel

        if self.transform:
            new_img = self.transform(new_img)

        if self.label_matrix is not None:
            tags = torch.tensor(self.label_matrix[idx])  # here, we will use only one label among multiple labels.
            # tags = torch.tensor(
            #     np.argmax(self.label_matrix[idx]))  # here, we will use only one label among multiple labels.
            return new_img, tags
        else:
            return new_img
