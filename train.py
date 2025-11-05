"""
Cassava Leaf Disease - Unified Training Script
----------------------------------------------
Trains one of the following models:
- ResNet50 with progressive unfreezing
- EfficientNet-B0 with gradual unfreezing and warmup scheduler
- Vision Transformer (ViT-B/16) with layer-wise LR decay

Usage:
    python train.py --model resnet
    python train.py --model efficientnet
    python train.py --model vit
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from models import (
    create_resnet50,
    create_efficientnet_b0,
    create_vit_b16,
    GradualWarmupScheduler,
    get_optimizer_params
)
from transformers import get_cosine_schedule_with_warmup


# ==========================================================
# Training and Validation Utilities
# ==========================================================
def mixup_data(x, y, alpha=1.0, device="cuda"):
    """Mixup augmentation."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, criterion, optimizer, device, scheduler=None, mixup=True):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(loader, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        if mixup:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, device=device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if mixup:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            true_labels = labels_a if lam > 0.5 else labels_b
        else:
            loss = criterion(outputs, labels)
            true_labels = labels

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(true_labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    return running_loss / len(loader), acc, macro_f1


def validate(model, loader, criterion, device):
    """Validate model performance."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    return running_loss / len(loader), acc, macro_f1


# ==========================================================
# Training Loop
# ==========================================================
def train_model(model_name: str, num_epochs: int = 15, batch_size: int = 32):
    """Main training routine supporting all three model types."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results/models", exist_ok=True)

    print(f"\n🚀 Training model: {model_name.upper()} on {device}\n")

    # Load data
    train_loader, val_loader, label_names, class_weights = get_dataloaders(
        dataset_path="data/cassava-leaf-disease-classification",
        batch_size=batch_size,
        use_weighted_sampler=True
    )
    num_classes = len(label_names)

    # Initialize model
    if model_name == "resnet":
        model = create_resnet50(num_classes).to(device)
        optimizer = optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = None  # Will define later in loop

    elif model_name == "efficientnet":
        model = create_efficientnet_b0(num_classes).to(device)
        params = [
            {"params": model.features.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-3}
        ]
        optimizer = optim.AdamW(params, weight_decay=0.01)
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - 3, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=8, warmup_epochs=3,
                                           total_epochs=num_epochs, after_scheduler=main_scheduler)

    elif model_name == "vit":
        model = create_vit_b16(num_classes).to(device)
        optimizer_params = get_optimizer_params(model, weight_decay=0.01, lr_init=1e-4)
        optimizer = optim.AdamW(optimizer_params)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_loader) * 2,
            num_training_steps=len(train_loader) * num_epochs
        )

    else:
        raise ValueError("Invalid model name. Choose from: resnet, efficientnet, vit")

    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Tracking
    best_val_f1 = 0
    train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = [], [], [], [], [], []

    # ======================== TRAINING ========================
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Progressive unfreezing (ResNet)
        if model_name == "resnet":
            if epoch == 0:
                for name, p in model.named_parameters():
                    p.requires_grad = "fc" in name
            elif epoch == 5:
                for name, p in model.named_parameters():
                    p.requires_grad = any(x in name for x in ["fc", "layer4"])
            elif epoch == 10:
                for name, p in model.named_parameters():
                    p.requires_grad = any(x in name for x in ["fc", "layer4", "layer3"])

        # Gradual unfreezing (EfficientNet)
        if model_name == "efficientnet":
            if epoch == 5:
                for p in model.features[-2:].parameters():
                    p.requires_grad = True
            if epoch == 10:
                for p in model.features[-4:].parameters():
                    p.requires_grad = True

        # Train and validate
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = f"results/models/{model_name}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model to {save_path}")

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        print(f"Train F1: {train_f1:.2f}%, Val F1: {val_f1:.2f}%")

    # ======================== PLOTTING ========================
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("Loss"); plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.title("Accuracy (%)"); plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_f1s, label="Train")
    plt.plot(val_f1s, label="Val")
    plt.title("Macro-F1 (%)"); plt.legend()

    plt.tight_layout()
    plt.savefig(f"results/{model_name}_training_curves.png")
    plt.close()

    print(f"\n🏁 Training complete for {model_name.upper()} | Best Val F1: {best_val_f1:.2f}%")
    print(f"Plots saved to results/{model_name}_training_curves.png")


# ==========================================================
# Main Entry Point
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cassava Disease Classification Models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet", "efficientnet", "vit"],
                        help="Model type to train")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train_model(model_name=args.model, num_epochs=args.epochs, batch_size=args.batch_size)
