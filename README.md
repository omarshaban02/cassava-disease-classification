# 🌿 Cassava Leaf Disease Classification

A deep learning project comparing **ResNet50**, **EfficientNet-B0**, and **Vision Transformer (ViT-B/16)** for cassava leaf disease detection using the [Kaggle Cassava Leaf Disease Dataset](https://www.kaggle.com/c/cassava-leaf-disease-classification).

## 📘 Overview

This repository provides a clean, modular PyTorch pipeline for training and evaluating multiple architectures on the cassava disease dataset.  
Each model is fine-tuned with advanced strategies such as:

- Progressive / gradual unfreezing  
- Mixup augmentation  
- Cosine warmup scheduling  
- Class-weighted loss for imbalance handling  

## 📂 Repository Structure

```plain

cassava-disease-classification/
│
├── data_loader.py          # Dataset + transforms + dataloaders
├── models.py               # ResNet50, EfficientNet-B0, ViT-B/16 definitions
├── train.py                # Unified training script
├── evaluate.py             # Evaluation and comparison
│
├── results/                # Trained models, metrics, and plots
│   ├── models/
│   ├── plots/
│   └── model_comparison_results.csv
│
├── requirements.txt
└── README.md

````

## ⚙️ Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/cassava-disease-classification.git
   cd cassava-disease-classification
    ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**

   - Download the Kaggle dataset:
     [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification)
   - Place files under:

     ```plain
     data/cassava-leaf-disease-classification/
     ├── train.csv
     └── train_images/
     ```

## 🧠 Train Models

Train any of the models with a single command:

```bash
python train.py --model resnet
python train.py --model efficientnet
python train.py --model vit
```

Options:

- `--epochs` → number of epochs (default: 15)
- `--batch_size` → batch size (default: 32)

Models and training plots are saved to:

```plain
results/models/
results/resnet_training_curves.png
```

## 📊 Evaluate Models

After training:

```bash
python evaluate.py
```

Outputs:

- Per-model metrics (Accuracy, Macro-F1, Inference Time)
- Confusion matrices
- Comparison CSV → `results/model_comparison_results.csv`
- Performance bar plot → `results/plots/model_comparison.png`

## 🧩 Key Features

- 🧠 ResNet50: progressive layer unfreezing
- ⚙️ EfficientNet-B0: gradual fine-tuning + warmup scheduler
- 🔬 ViT-B/16: transformer-based classifier with layer-wise LR decay
- 🧪 Mixup augmentation & class-balanced loss
- 📈 Automatic result visualization and comparison

## 📄 Requirements

```plain
torch
torchvision
transformers
tqdm
pandas
numpy
matplotlib
seaborn
scikit-learn
Pillow
```

## 🏁 Results Summary

| Model           | Accuracy (%) | Macro-F1 (%) | Time per Sample (s) |
| --------------- | ------------ | ------------ | ------------------- |
| ResNet50        | —            | —            | —                   |
| EfficientNet-B0 | —            | —            | —                   |
| ViT-B/16        | —            | —            | —                   |

(*Populated automatically after running `evaluate.py`*)

## 🧾 License

Released under the **MIT License** – free for research and educational use.

## 👤 Author

Developed by **Omar Shaban** \
Biomedical & AI Engineer \
<omar.an.shaban@gmail.com>
