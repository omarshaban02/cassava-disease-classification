"""
Cassava Leaf Disease - Model Evaluation and Comparison
------------------------------------------------------
Evaluates all trained models (ResNet50, EfficientNet-B0, ViT-B/16)
and compares their performance on the validation set.

Outputs:
- Accuracy, Macro-F1, and Inference Time per model
- Confusion matrices and comparison plots
- CSV summary: results/model_comparison_results.csv
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from data_loader import get_dataloaders
from models import create_resnet50, create_efficientnet_b0, create_vit_b16


# ==========================================================
# Evaluation Function
# ==========================================================
@torch.no_grad()
def evaluate_model(model, loader, device):
    """Compute accuracy, macro-F1, and inference time."""
    model.eval()
    all_preds, all_labels = [], []
    start_time = time.time()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    duration = time.time() - start_time
    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    return acc, macro_f1, duration / len(loader.dataset), all_preds, all_labels


def plot_confusion_matrix(labels, preds, label_names, model_name):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names.values(),
                yticklabels=label_names.values())
    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs("results/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"results/plots/{model_name}_confusion_matrix.png")
    plt.close()


# ==========================================================
# Main Evaluation Routine
# ==========================================================
def evaluate_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Evaluating models on {device}")

    # Load data
    _, val_loader, label_names, _ = get_dataloaders(
        dataset_path="data/cassava-leaf-disease-classification",
        batch_size=32,
        use_weighted_sampler=False
    )
    num_classes = len(label_names)

    model_constructors = {
        "resnet": create_resnet50,
        "efficientnet": create_efficientnet_b0,
        "vit": create_vit_b16
    }

    results = []

    for model_name, constructor in model_constructors.items():
        model_path = f"results/models/{model_name}_best.pth"
        if not os.path.exists(model_path):
            print(f"⚠️  Skipping {model_name.upper()} (no saved model found)")
            continue

        print(f"\n📦 Loading {model_name.upper()}...")
        model = constructor(num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        acc, macro_f1, avg_time, preds, labels = evaluate_model(model, val_loader, device)

        print(f"✅ {model_name.upper()} | Acc: {acc:.2f}% | Macro-F1: {macro_f1:.2f}% | Time/sample: {avg_time:.4f}s")
        results.append([model_name, acc, macro_f1, avg_time])

        # Confusion matrix
        plot_confusion_matrix(labels, preds, label_names, model_name)

    # Save comparison CSV
    os.makedirs("results", exist_ok=True)
    df_results = pd.DataFrame(results, columns=["Model", "Accuracy (%)", "Macro-F1 (%)", "Time per sample (s)"])
    df_results.to_csv("results/model_comparison_results.csv", index=False)
    print("\n📊 Results saved to results/model_comparison_results.csv")

    # Plot comparison
    if len(results) > 1:
        df_results_melted = df_results.melt(id_vars="Model",
                                            value_vars=["Accuracy (%)", "Macro-F1 (%)"],
                                            var_name="Metric",
                                            value_name="Score")
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df_results_melted, x="Model", y="Score", hue="Metric")
        plt.title("Model Performance Comparison")
        plt.tight_layout()
        plt.savefig("results/plots/model_comparison.png")
        plt.close()
        print("📈 Saved bar plot: results/plots/model_comparison.png")


# ==========================================================
# Entry Point
# ==========================================================
if __name__ == "__main__":
    evaluate_all_models()
