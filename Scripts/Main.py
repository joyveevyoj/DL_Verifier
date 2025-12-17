"""
Main entry point for training.
Loads configuration and starts training.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Scripts.config_loader import load_config
from Scripts.Train_bce import train_bce
from Scripts.Train_pAUC import train_pAUC


def main():
    """Main function to load config and start training."""
    # Load configuration
    config = load_config('configure.yaml')
    
    bce_cfg = config.BCE_TRAIN
    pauc_cfg = config.PAUC_TRAIN
    parser = argparse.ArgumentParser(description="Verifier training")
    parser.add_argument(
        "--mode",
        choices=["bce", "pauc", "both"],
        default="bce",
        help="Which training flow to run"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Configuration Loaded")
    print("=" * 60)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    if args.mode in ("bce", "both"):
        print("-" * 60)
        print("BCE Parameters:")
        print(f"  Batch Size: {bce_cfg.BATCH_SIZE}")
        print(f"  Epochs: {bce_cfg.EPOCH_NUM}")
        print(f"  LR: {bce_cfg.LEARNING_RATE}")
        print(f"  Output Dir: {bce_cfg.OUTPUT_DIR}")
    if args.mode in ("pauc", "both"):
        print("-" * 60)
        print("pAUC Parameters:")
        print(f"  Batch Size: {pauc_cfg.BATCH_SIZE}")
        print(f"  Epochs: {pauc_cfg.EPOCH_NUM}")
        print(f"  LR: {pauc_cfg.LEARNING_RATE}")
        print(f"  Output Dir: {pauc_cfg.OUTPUT_DIR}")
    print("-" * 60)
    print(f"Train Dataset: {config.TRAIN_DATASET_PATH}")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Verifier training")
    parser.add_argument(
        "--mode",
        choices=["bce", "pauc", "both"],
        default="bce",
        help="Which training flow to run"
    )
    args = parser.parse_args()

    # Load dataset
    print(f"\nLoading dataset from {config.TRAIN_DATASET_PATH}...")
    with open(config.TRAIN_DATASET_PATH, 'r', encoding='utf-8') as f:
        raw_questions = json.load(f)
    
    print(f"Loaded {len(raw_questions)} samples")
    if bce_cfg.DEBUG_SAMPLE_SIZE:
        print(f"Using DEBUG mode: {bce_cfg.DEBUG_SAMPLE_SIZE} samples")

    if args.mode in ("bce", "both"):
        train_bce(config, raw_questions)
    if args.mode in ("pauc", "both"):
        train_pAUC(config, raw_questions)


if __name__ == '__main__':
    main()

