import torch
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding
from libauc.metrics import pauc_roc_score

# Assuming these modules are available in the current directory
from verifier_dataset import VerifierDataset
from lora_model import build_bce_model

import warnings
import logging
from transformers import logging as hf_logging
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Configuration Section (Updated based on your provided config) ---
class EvalConfig:
    # Base model name
    MODEL_NAME = "Qwen/Qwen3-0.6B"

    # Updated Max Length and Batch Size
    MAX_LENGTH = 256
    BATCH_SIZE = 32

    # Updated LoRA parameters
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1

    # Path to the test dataset (Assuming relative path is same as before)
    TEST_DATA_PATH = "data/verifier_dataset_test.json"

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Registry ---
# Format: "Model ID": {"path": "Checkpoint Path", "pauc_params": {pAUC calculation params}}
# This defines which models to evaluate and their specific target metrics
MODEL_REGISTRY = {
    "bce": {
        "path": "checkpoints/bce/bce_best_model.pt",
        "pauc_params": {"max_fpr": 0.3} # BCE default is FPR<=0.3
    },
    "sopas_fpr0.3": {
        "path": "checkpoints/sopas_fpr0.3/pauc_best_model.pt",
        "pauc_params": {"max_fpr": 0.3}
    },
    "sopas_fpr0.5": {
        "path": "checkpoints/sopas_fpr0.5/pauc_best_model.pt",
        "pauc_params": {"max_fpr": 0.5}
    },
    "sotas_tpr0.6_fpr0.4": {
        "path": "checkpoints/sotas_tpr0.6_fpr0.4/pauc_best_model.pt",
        "pauc_params": {"min_tpr": 0.6, "max_fpr": 0.4}
    },
    "sotas_tpr0.5_fpr0.5": {
        "path": "checkpoints/sotas_tpr0.5_fpr0.5/pauc_best_model.pt",
        "pauc_params": {"min_tpr": 0.5, "max_fpr": 0.5}
    }
}

def load_data(file_path):
    print(f"Loading test data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples.")
    return data

def evaluate_model(model_name, config_entry, test_loader, tokenizer, base_config):
    """
    Load a single model and perform inference evaluation on the test set.
    """
    ckpt_path = config_entry["path"]
    pauc_params = config_entry["pauc_params"]

    print(f"\n{'='*20}")
    print(f"Evaluating Model: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Target Metrics: {pauc_params}")
    print(f"{'='*20}")

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}. Skipping.")
        return None

    # 1. Build model structure
    # Utilizing the build function from lora_model.py
    # Note: We use the updated LoRA params (R=8, Alpha=16) here
    model = build_bce_model(
        model_name=base_config.MODEL_NAME,
        lora_r=base_config.LORA_R,
        lora_alpha=base_config.LORA_ALPHA,
        lora_dropout=base_config.LORA_DROPOUT,
        pad_token_id=tokenizer.pad_token_id,
        config=None
    )

    # 2. Load weights
    print("Loading weights...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=base_config.DEVICE)
        # Handle potential DDP wrapper (if 'module.' prefix exists) or load directly
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None

    model.to(base_config.DEVICE)
    model.eval()

    # 3. Inference loop
    all_preds = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Inf {model_name}"):
            input_ids = batch['input_ids'].to(base_config.DEVICE)
            attention_mask = batch['attention_mask'].to(base_config.DEVICE)
            labels = batch['labels'].float().cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            probs = torch.sigmoid(logits).float().cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. Calculate metrics
    # Accuracy
    binary_preds = (all_preds > 0.5).astype(int)
    accuracy = (binary_preds == all_labels).mean()

    # pAUC (based on specific parameters)
    try:
        score_pauc = pauc_roc_score(all_labels, all_preds, **pauc_params)
    except Exception as e:
        print(f"Warning: pAUC calculation failed ({e}). Setting to 0.")
        score_pauc = 0.0

    # Calculate a standard pAUC (FPR<=0.3) for reference/comparison
    try:
        std_pauc = pauc_roc_score(all_labels, all_preds, max_fpr=0.3)
    except:
        std_pauc = 0.0

    result = {
        "model": model_name,
        "acc": accuracy,
        "target_pauc": score_pauc,
        "target_params": str(pauc_params),
        "std_pauc_0.3": std_pauc
    }

    print(f"Result for {model_name}: Acc={accuracy:.2%}, Target pAUC={score_pauc:.4f}")

    # Clear memory
    del model
    torch.cuda.empty_cache()

    return result

def main():
    config = EvalConfig()

    # 1. Prepare data
    if not os.path.exists(config.TEST_DATA_PATH):
        print(f"Test data not found at {config.TEST_DATA_PATH}")
        return

    raw_data = load_data(config.TEST_DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_dataset = VerifierDataset(raw_data, tokenizer, config.MAX_LENGTH)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )

    # 2. Evaluate all models sequentially
    results = []
    for model_name, model_conf in MODEL_REGISTRY.items():
        res = evaluate_model(model_name, model_conf, test_loader, tokenizer, config)
        if res:
            results.append(res)

    # 3. Summary Report
    print("\n\n" + "="*60)
    print(f"{'FINAL EVALUATION REPORT':^60}")
    print("="*60)
    # Print Table Header
    print(f"{'Model Name':<25} | {'Acc':<8} | {'Target pAUC':<12} | {'Params':<25}")
    print("-" * 75)

    for res in results:
        print(f"{res['model']:<25} | {res['acc']:.2%}   | {res['target_pauc']:.4f}       | {res['target_params']:<25}")
    print("="*60)

if __name__ == "__main__":
    main()
