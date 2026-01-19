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

# --- Suppress Warnings ---
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Configuration Section ---
class EvalConfig:
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    MAX_LENGTH = 256
    BATCH_SIZE = 32
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    TEST_DATA_PATH = "data/verifier_dataset_test.json"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Metrics Definition ---
EVAL_METRICS = {
    "FPR<=0.3": {"max_fpr": 0.3},
    "FPR<=0.5": {"max_fpr": 0.5},
    "TPR>=0.6|FPR<=0.4": {"min_tpr": 0.6, "max_fpr": 0.4},
    "TPR>=0.5|FPR<=0.5": {"min_tpr": 0.5, "max_fpr": 0.5}
}

# --- Model Registry ---
MODEL_REGISTRY = {
    "BCE (Baseline)": "checkpoints/bce/bce_best_model.pt",
    "SOPA (FPR 0.3)": "checkpoints/sopas_fpr0.3/pauc_best_model.pt",
    "SOPA (FPR 0.5)": "checkpoints/sopas_fpr0.5/pauc_best_model.pt",
    "SOTA (TPR 0.6)": "checkpoints/sotas_tpr0.6_fpr0.4/pauc_best_model.pt",
    "SOTA (TPR 0.5)": "checkpoints/sotas_tpr0.5_fpr0.5/pauc_best_model.pt"
}

def load_data(file_path):
    print(f"Loading test data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples.")
    return data

def evaluate_model(model_name, ckpt_path, test_loader, tokenizer, base_config):
    """
    Load a model and calculate all defined metrics.
    """
    print(f"\n{'='*20}")
    print(f"Evaluating Model: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*20}")

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}. Skipping.")
        return None

    # 1. Build model
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
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None

    model.to(base_config.DEVICE)
    model.eval()

    # 3. Inference
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

    # 4. Calculate All Metrics
    binary_preds = (all_preds > 0.5).astype(int)
    accuracy = (binary_preds == all_labels).mean()

    metrics_results = {}
    for metric_name, params in EVAL_METRICS.items():
        try:
            score = pauc_roc_score(all_labels, all_preds, **params)
        except Exception as e:
            score = 0.0
        metrics_results[metric_name] = score

    result = {
        "model": model_name,
        "acc": accuracy,
        "metrics": metrics_results
    }

    # Print immediate result summary (5 decimal places)
    print(f"Result for {model_name}: Acc={accuracy:.5f}")
    for k, v in metrics_results.items():
        print(f"  - {k}: {v:.5f}")

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

    # 2. Evaluate all models
    results = []
    for model_name, ckpt_path in MODEL_REGISTRY.items():
        res = evaluate_model(model_name, ckpt_path, test_loader, tokenizer, config)
        if res:
            results.append(res)

    # 3. Summary Report
    print("\n\n" + "="*120)
    print(f"{'FINAL EVALUATION REPORT':^120}")
    print("="*120)

    # Define column widths
    name_w = 25
    acc_w = 10
    metric_w = 18

    # Build Header
    header = f"{'Model Name':<{name_w}} | {'Acc':<{acc_w}}"
    metric_keys = list(EVAL_METRICS.keys())
    for k in metric_keys:
        header += f" | {k:<{metric_w}}"

    print(header)
    print("-" * len(header))

    # Print Rows
    for res in results:
        # Changed format from .2% to .5f
        row = f"{res['model']:<{name_w}} | {res['acc']:.5f}   "
        for k in metric_keys:
            val = res['metrics'].get(k, 0.0)
            # Changed format from .4f to .5f
            row += f" | {val:.5f}".ljust(metric_w + 3)
        print(row)

    print("="*120)

if __name__ == "__main__":
    main()