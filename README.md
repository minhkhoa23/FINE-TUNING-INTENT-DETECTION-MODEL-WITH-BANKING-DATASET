# Fine-Tuning Intent Detection Model with Banking Dataset

This repository provides scripts and configurations to fine-tune an intent detection model using a banking dataset. Follow the steps below to set up the environment, preprocess data, train the model, and run inference.

Student Information: \
Full name: Thái Minh Khoa \
Student ID: 23127394

---
## Link video demo in Google drive
https://drive.google.com/file/d/169GsFWYSxugPyZKMCEZ9ssgH8FEv22DZ/view?usp=sharing

---
## Link notebook Kaggle
https://www.kaggle.com/code/minhkhoathi/nlp-doanh-nghiep-lab2

---
## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Dataset Preparation and Preprocessing](#dataset-preparation-and-preprocessing)
3. [Training the Model](#training-the-model)
4. [Evaluating the Model](#evaluating-the-model)
5. [Running Inference](#running-inference)

---

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- GPU with CUDA support (optional but recommended)

### Install Dependencies

1. Clone the repository:
   ```bash
   !git clone https://github.com/minhkhoa23/FINE-TUNING-INTENT-DETECTION-MODEL-WITH-BANKING-DATASET.git
   ```

2. Install the required Python packages:
   ```bash
   %cd FINE-TUNING-INTENT-DETECTION-MODEL-WITH-BANKING-DATASET
   !pip install -r requirements.txt

   !pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
   ```

---

## Dataset Preparation and Preprocessing

To preprocess the data, use the following command:

```bash
!python scripts/preprocess_data.py \
  --train_path sample_data/train.csv \
  --test_path sample_data/test.csv \
  --output_dir sample_data/processed \
  --train_fraction 0.5 \
  --test_fraction 0.5 \
  --create_val \
  --val_size 0.1 \
  --lowercase
```

### Arguments:
- `--train_path`: Path to the training dataset (default: `sample_data/train.csv`).
- `--test_path`: Path to the testing dataset (default: `sample_data/test.csv`).
- `--output_dir`: Directory to save the processed data (default: `sample_data/processed`).
- `--train_fraction`: Fraction of the training data to use (default: `0.5`).
- `--test_fraction`: Fraction of the testing data to use (default: `0.5`).
- `--create_val`: Flag to create a validation set.
- `--val_size`: Size of the validation set as a fraction of the training data (default: `0.1`).
- `--lowercase`: Flag to convert all text to lowercase.

Ensure that the `sample_data/` directory contains the `train.csv` and `test.csv` files before running the script.

---

## Training the Model

To train the model, use the `train.sh` script:

```bash
!python scripts/train.py [CONFIG_PATH]
```

- `CONFIG_PATH`: Path to the training configuration file (default: `configs/train.yaml`).

Example:
```bash
!python scripts/train.py --config configs/train.yaml
```

---

## Evaluating the Model

To evaluate the model, use the `evaluate.sh` script:

```bash
!python scripts/evaluate.py [CONFIG_PATH] [TEST_PATH] [OUTPUT_DIR]
```

- `CONFIG_PATH`: Path to the evaluation configuration file (default: `configs/inference.yaml`).
- `TEST_PATH`: Path to the processed test dataset (default: `sample_data/processed/test.csv`).
- `OUTPUT_DIR`: Directory to save evaluation results (default: `outputs/eval_results`).

Example:
```bash
!python scripts/evaluate.py \
  --config configs/inference.yaml \
  --test_path sample_data/processed/test.csv \
  --output_dir outputs/eval_results
```

---

## Running Inference

To run inference, use the `inference.sh` script:

```bash
!bash inference.sh [CONFIG_PATH] [INPUT_TEXT]
```

- `CONFIG_PATH`: Path to the inference configuration file (default: `configs/inference.yaml`).
- `INPUT_TEXT`: Input text for intent detection (default: `"I lost my card"`).

Example:
```bash
!bash inference.sh configs/inference.yaml "I lost my card"
```

---

## Notes
- Ensure that the configuration files in the `configs/` directory are properly set up before running the scripts.
- For GPU usage, ensure that CUDA is installed and available on your system.

---