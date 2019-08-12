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
import random
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
                     num_workers=0,
                     infer_batch_size=32,
                     transform=None,
                     infer_transform=None,
                     val_ratio=0.1,
                     use_random_label=False, seed=42
                     ):
    image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images')
    label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
    train_meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')
    # tags_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'tags.tsv')
    meta_data = pd.read_csv(train_meta_path, delimiter=',', header=0)
    # tags_meta_data = pd.read_csv(tags_path, delimiter='\t', header=0)
    # print(tags_meta_data.iloc[:30])

    label_matrix = np.load(label_path)

    if transform is None:
        transform = transforms.Compose(
                          [transforms.Resize((input_size, input_size)), transforms.ToTensor()])

    if val_ratio > 0:
        rs = ShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        for train_index, val_index in rs.split(meta_data):
            val_meta_data = meta_data.iloc[val_index]
            train_meta_data = meta_data.iloc[train_index]

            train_labels = label_matrix[train_index]
            val_labels = label_matrix[val_index]

            print("train_x", len(train_meta_data))
            print("val_x", len(val_meta_data))
            print("train_y", len(train_labels))
            print("val_y", len(val_labels))

        if infer_transform is None:
            infer_transform = transforms.Compose(
                [transforms.Resize((input_size, input_size)), transforms.ToTensor()])

        train_dataloader = DataLoader(
            AIRushDataset(image_dir, train_meta_data, label_path=train_labels,
                          transform=transform,use_random_label=use_random_label),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)

        val_dataloader = DataLoader(
            AIRushDataset(image_dir, val_meta_data, label_path=val_labels,
                          transform=infer_transform),
            batch_size=infer_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True)
        return train_dataloader, val_dataloader
    else:
        train_dataloader = DataLoader(
            AIRushDataset(image_dir, meta_data, label_path=label_matrix,
                          transform=transform, use_random_label=use_random_label),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)
        return train_dataloader, None


class AIRushDataset(Dataset):
    def __init__(self, image_data_path, meta_data, label_path=None, transform=None, use_random_label=False, seed=42):
        self.meta_data = meta_data
        self.image_dir = image_data_path
        self.label_matrix = label_path
        self.transform = transform
        self.use_random_label = use_random_label
        self.seed = seed

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

            if self.use_random_label:
                random.seed(self.seed)
                tags = np.zeros_like(self.label_matrix[idx])
                tags[random.choice(np.where(self.label_matrix[idx] == 1)[0])] = 1
            else:
                tags = torch.tensor(self.label_matrix[idx])  # here, we will use only one label among multiple labels.
            # tags = torch.tensor(
            #     np.argmax(self.label_matrix[idx]))  # here, we will use only one label among multiple labels.
            return new_img, tags
        else:
            return new_img
