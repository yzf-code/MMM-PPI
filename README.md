# MMM-PPI

## 📖 Overview

This is the repository for 'Enhancing Protein-Protein Interaction Prediction with Hierarchical Motif-based Multimodal Protein Embedding'.

## 📁 Repository Structure

```
MMM-PPI/
├── ckpt/                          # Pre-trained model checkpoints (download from HuggingFace)
├── data/                          # Dataset files (download from HuggingFace)
├── motif_detection/               # Motif detection scripts and pre-computed results
│   └── README.md                  # Instructions for running motif detection from scratch
├── Classifier_model.py            # PPI classifier model
├── dataloader.py                  # Data loading and preprocessing
├── environment.yml                # Conda environment specification
├── main.py                        # Main entry point for training/evaluation
├── pretrain_Pair_wise_Encoder.py  # Pair-wise encoder pre-training 
├── run.sh                         # Example shell script for running experiments
└── utils.py                       # Utility functions
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yzf-code/MMM-PPI.git
cd MMM-PPI
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate MMM-PPI
```

## 📦 Data & Checkpoints

We host the processed datasets and pre-trained checkpoints on HuggingFace:

🔗 **HuggingFace Repository:** https://huggingface.co/datasets/yzf1102/MMM-PPI

### Download instructions

bash

```bash
# Option 1: using huggingface_hub
pip install huggingface_hub
huggingface-cli download yzf1102/MMM-PPI --repo-type dataset --local-dir ./hf_assets

# Option 2: using git lfs
git lfs install
git clone https://huggingface.co/datasets/yzf1102/MMM-PPI hf_assets
```

After downloading, organize the files as follows:

```
MMM-PPI/
├── ckpt/
├── data/
└── main.py
└── run.sh
└── ...
```

> 💡 Make sure the `ckpt/` and `data/` folders are placed at the **root** of the repository, matching the structure shown above.

## 🧬 Motif Detection

We provide **pre-computed motif detection results** in the huggface, so you can run the main pipeline directly without re-processing.

If you would like to **run motif detection from scratch** on your own data, please refer to:

```
motif_detection/README.md
```

for detailed instructions.

## 🏃 Usage

The simplest way to reproduce our experiments is via the provided shell script:

```bash
bash run.sh
```

You can modify the arguments inside `run.sh` to fit your own experimental settings.

