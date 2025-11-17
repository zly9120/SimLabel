## Multi-Annotator Learning with missing labels

This repository contains the implementation of our SimLabel approach based on QuMATL backbone architecture, providing core logic and essential components to support the reproducibility of our research findings.
The organized formal codes will be released upon acceptance.

## 📁 Structure

```
├── simlabel_amer.py          # Core training/testing code for AMER dataset
├── simlabel_street.py        # Core training/testing code for STREET dataset  
├── process_dataset_mer2.py   # Splitting AMER2 dataset 
├── environment.yml           # Conda environment configuration
└── README.md                 # This file
```

## 🛠️ Environment Setup

### Prerequisites
- Anaconda or Miniconda
- Python 3.8+
### Installation
conda env create -f environment.yml

## 🚀 Usage

### Training AMER
python simlabel_amer.py --batch_size 8 --epochs 200 --confidence_threshold 0.6
### Testing AMER
python simlabel_amer.py --evaluate --checkpoint_name best_model.pth

### Training STREET with 40% missing rate for Safe perspective
python simlabel_street.py --category Safe --missing_rate 0.4 --batch_size 64 --epochs 200

### Training STREET for other perspectives
python simlabel_street.py --category Happiness --missing_rate 0.4
python simlabel_street.py --category Healthy --missing_rate 0.4

### Evaluation STREET
python simlabel_street.py --evaluate --category Safe --checkpoint_name best_model.pth

## 🔬 Reproducibility

1. Use the provided environment: Always run experiments using the `environment.yml` configuration
2. Fixed random seeds: All random operations use predefined seeds
