# DL_Verifier

Test-time scaling experiment for LLM verification using Qwen3-0.6B on GSM8K dataset.

## Project Structure (Scripts)

The core logic of this project is located in the `Scripts/` directory:

- `train_one_head_roc.py`: Main entry point for Stage 1 training (BCE/pAUC).
- `train_two_head.py`: Main entry point for Stage 2 training (Difficulty Prediction).
- `Verification_strategy.py`: Evaluation script for comparing verification strategies (Best-of-N, Adaptive N, etc.).
- `two_head_model.py`: Model architecture for the two-head verifier (Classification + Difficulty).
- `lora_model.py`: Model building logic for Stage 1 verifiers using LoRA adapters.
- `train_bce.py`: Implementation of the Binary Cross-Entropy training loop.
- `train_pauc.py`: Implementation of the Partial AUC (pAUC) optimization loop.
- `verifier_dataset.py`: PyTorch Dataset classes for training and evaluation.
- `verifier_dataset_generation.py`: Script to generate training samples from LLM responses.
- `config.yaml`: Central configuration for model names, paths, and hyperparameters.
- `config_loader.py`: Utility to load and parse the configuration file.

## Environment Setup

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate llm-verifier
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\scripts\activate
pip install -r requirement.txt
```

### For CUDA (>= 12.1) (Recommended)
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### For MacOS (Not Recommended)
```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
```

## Training and Evaluation

### 0. Data Preparation
Before running any training scripts, ensure your dataset is ready. If you have the training data in a compressed format, unzip it into the `data/` folder:
```bash
# Example (if applicable)
unzip data/verifier_dataset_train.zip -d data/
```
Ensure `data/verifier_dataset_train.json` and `data/verifier_dataset_test.json` are present.

### 1. First Stage Training One-Head Verifier (ROC/pAUC)
To train the binary classification verifier optimizing for the AUC-ROC metric:
```bash
python Scripts/train_one_head_roc.py --mode bce
```
**Arguments:**
- `--mode`: Which training flow to run. Options:
    - `bce` (default): Trains for binary classification and saves `bce_best_model.pt`.
    - `pauc`: Trains for partial AUC optimization and saves `pauc_best_model.pt`.
    - `both`: Runs both training flows sequentially and saves both checkpoints.

### 2. Second Stage Training Two-Head Model (Difficulty Prediction)
Stage 2 training for the difficulty prediction head (Head B) using the pre-trained verifier (Head A). This script uses settings from the `TWO_HEAD_TRAIN` section in `config.yaml`.
```bash
python Scripts/train_two_head.py
```
**Notes:**
- It loads a pre-trained BCE model (specified by `START_FROM_CHECKPOINT` in `config.yaml`) and trains a second head for difficulty prediction.
- Saved checkpoint: `checkpoints/two_head/two_head_best.pt`.

### 3. Verification Strategy & Comparison
To evaluate and compare different verification strategies (Best-of-N, Rejection Sampling, Adaptive N):
```bash
# Example running comparison with the two-head model
python Scripts/Verification_strategy.py --verifier two_head --n 10 --best_n 10 --threshold 0.5 --lambd 0.05
```
**Arguments:**
- `--verifier`: Choose between `bce`, `pauc`, or `two_head`.
- `--n`: Number of questions from the test set to evaluate.
- `--best_n`: The fixed $N$ budget for Best-of-N and Rejection Sampling.
- `--threshold`: The confidence threshold (0-1) for Rejection Sampling.
- `--lambd`: The cost parameter $\lambda$ for Adaptive N (lower $\lambda$ = higher $N^*$).

