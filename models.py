"""
Cassava Leaf Disease - Model Architectures
------------------------------------------
Defines and initializes the following models:
- ResNet50 with custom classifier
- EfficientNet-B0 with enhanced regularization
- Vision Transformer (ViT-B/16) with custom head
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


# ==========================================================
# 1. ResNet50
# ==========================================================
def create_resnet50(num_classes: int):
    """
    Create ResNet50 model with custom classifier and dropout layers.
    Progressive unfreezing can be handled externally during training.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model


# ==========================================================
# 2. EfficientNet-B0
# ==========================================================
class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Gradual Warmup Scheduler as used in EfficientNet training.
    """
    def __init__(self, optimizer, multiplier, warmup_epochs, total_epochs, after_scheduler=None):
        self.multiplier = multiplier
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.after_scheduler = after_scheduler
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epochs + 1.)
                    for base_lr in self.base_lrs]
        else:
            if self.after_scheduler:
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self.last_epoch < self.warmup_epochs:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            if self.after_scheduler:
                self.after_scheduler.step(epoch - self.warmup_epochs)


def create_efficientnet_b0(num_classes: int):
    """
    Create EfficientNet-B0 with modified classifier head.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Modify classifier with dropout + batch normalization
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(1280, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )

    return model


# ==========================================================
# 3. Vision Transformer (ViT-B/16)
# ==========================================================
def create_vit_b16(num_classes: int):
    """
    Create and initialize a Vision Transformer (ViT-B/16) model with a custom head.
    """
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    # Modify classification head
    num_features = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.LayerNorm(num_features),
        nn.Dropout(0.2),
        nn.Linear(num_features, 512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, num_classes)
    )

    return model


def get_optimizer_params(model, weight_decay: float = 0.01, lr_init: float = 1e-4):
    """
    Layer-wise learning rate decay for ViT.
    Higher LR for classification head.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and "heads" not in n],
            "weight_decay": weight_decay,
            "lr": lr_init
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and "heads" not in n],
            "weight_decay": 0.0,
            "lr": lr_init
        },
        {
            "params": [p for n, p in param_optimizer if "heads" in n],
            "weight_decay": weight_decay,
            "lr": lr_init * 10
        },
    ]
    return optimizer_parameters


if __name__ == "__main__":
    # Simple test to verify model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 5
    resnet = create_resnet50(num_classes).to(device)
    efficientnet = create_efficientnet_b0(num_classes).to(device)
    vit = create_vit_b16(num_classes).to(device)

    print("✅ Models initialized successfully:")
    print(f"ResNet50 parameters: {sum(p.numel() for p in resnet.parameters()) / 1e6:.2f}M")
    print(f"EfficientNet-B0 parameters: {sum(p.numel() for p in efficientnet.parameters()) / 1e6:.2f}M")
    print(f"ViT-B/16 parameters: {sum(p.numel() for p in vit.parameters()) / 1e6:.2f}M")
