import json
import torch

import sys
import random
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import types
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
# Create dataloaders
from transformers import DataCollatorWithPadding
from torch.utils.tensorboard import SummaryWriter

# Construct path relative to project root
_cwd = os.getcwd()
if os.path.basename(_cwd) == 'scripts':
    # If we're in scripts folder, go up one level
    DATASET_PATH = os.path.join(os.path.dirname(_cwd), "data", "verifier_dataset_train.json")
    OUTPUT_DIR = os.path.join(os.path.dirname(_cwd), "outputs")
elif os.path.basename(_cwd) == 'notebooks':
    # If we're in notebooks folder, go up two levels
    DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(_cwd)), "data", "verifier_dataset_train.json")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(_cwd)), "outputs")
else:
    # If we're in project root
    DATASET_PATH = os.path.join(_cwd, "data", "verifier_dataset_train.json")
    OUTPUT_DIR = os.path.join(_cwd, "outputs")

MODEL_NAME = "Qwen/Qwen3-0.6B"





def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running on device: {device}")


if __name__ == "__main__":
    main()