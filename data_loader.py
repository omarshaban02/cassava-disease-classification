"""
Cassava Leaf Disease - Data Loader
----------------------------------
This module handles:
- Loading the Cassava dataset metadata (train.csv)
- Custom Dataset class
- Data augmentation and normalization
- Stratified train/validation split
- Weighted sampling for class imbalance
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import torch


class CassavaDataset(Dataset):
    """Custom dataset for Cassava Leaf Disease Classification."""

    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id']
        label = int(self.df.iloc[idx]['label'])

        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(dataset_path: str = "data/cassava-leaf-disease-classification",
                    batch_size: int = 32,
                    val_split: float = 0.2,
                    num_workers: int = 2,
                    use_weighted_sampler: bool = True):
    """
    Prepare DataLoaders for training and validation.
    Returns:
        train_loader, val_loader, label_names, class_weights
    """

    dataset_path = Path(dataset_path)
    csv_path = dataset_path / "train.csv"
    img_dir = dataset_path / "train_images"

    # Load dataset metadata
    train_df = pd.read_csv(csv_path)

    # Label mapping
    label_names = {
        0: "Cassava Bacterial Blight (CBB)",
        1: "Cassava Brown Streak Disease (CBSD)",
        2: "Cassava Green Mottle (CGM)",
        3: "Cassava Mosaic Disease (CMD)",
        4: "Healthy"
    }
    train_df['disease_name'] = train_df['label'].map(label_names)

    # Split into train and validation
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_split,
        stratify=train_df['label'],
        random_state=42
    )

    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # Create datasets
    train_dataset = CassavaDataset(train_df, img_dir, transform=data_transforms['train'])
    val_dataset = CassavaDataset(val_df, img_dir, transform=data_transforms['val'])

    # Compute class weights for balancing
    classes = np.unique(train_df['label'])
    class_weights_np = compute_class_weight(class_weight='balanced',
                                            classes=classes,
                                            y=train_df['label'].values)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)

    # Weighted sampling (optional)
    if use_weighted_sampler:
        sample_weights = [class_weights_np[label] for label in train_df['label'].tolist()]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Class weights: {class_weights_np}")

    return train_loader, val_loader, label_names, class_weights


if __name__ == "__main__":
    # Example usage for quick testing
    train_loader, val_loader, label_names, class_weights = get_dataloaders(
        dataset_path="data/cassava-leaf-disease-classification",
        batch_size=16
    )
    images, labels = next(iter(train_loader))
    print(f"Sample batch shape: {images.shape}, labels: {labels[:8]}")
