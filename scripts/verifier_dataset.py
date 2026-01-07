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


class VerifierDataset(Dataset):
    def __init__(self, raw_data_list, tokenizer, max_length=512):
        self.samples = []
        for entry in raw_data_list:
            question = entry['question']
            answers = entry['answers']
            labels = entry['answer_labels']
            ref_answer = entry["reference_answer"]

            # If a reference answer exists, append it to the end and add the corresponding label 1
            if ref_answer is not None:
                answers = answers + [ref_answer]
                labels = labels + [1]
            for ans, label in zip(answers, labels):
                text = f"Question: {question}\nAnswer: {ans}"
                # Label must be float for BCE Loss (0.0 or 1.0)
                self.samples.append({"text": text, "label": float(label)})
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encodings = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False 
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(item["label"], dtype=torch.float),
            "index": torch.tensor(idx, dtype=torch.long)        }


class Adaptive_N_VerifierDataset(Dataset):
    def __init__(self, raw_data_list, tokenizer, max_length=512):
        self.samples = []
        for entry in raw_data_list:
            question = entry['question']
            correct_answers_num = entry['correct_answers_num']
            total_answers_num = entry['total_answers_num']
            
            # Compute empirical probability (correct rate)
            empirical_p = correct_answers_num / total_answers_num if total_answers_num > 0 else 0.0
            
            # Only use the question, no answer/solution
            text = f"Question: {question}"
            
            # Label is the empirical probability (float between 0.0 and 1.0)
            self.samples.append({"text": text, "label": float(empirical_p)})
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encodings = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False 
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(item["label"], dtype=torch.float)
        }


# difficultity classification dataset
def probability_to_difficulty_class(prob, num_classes):
    """
    Map a probability value to a difficulty class.
    
    Args:
        prob: float between 0.0 and 1.0 (empirical correct rate)
        num_classes: int, number of difficulty levels
    
    Returns:
        int, class index from 0 to num_classes-1
        
    Examples:
        num_classes=2: [0.0, 0.5) → 0, [0.5, 1.0] → 1
        num_classes=3: [0.0, 0.33) → 0, [0.33, 0.67) → 1, [0.67, 1.0] → 2
        num_classes=4: [0.0, 0.25) → 0, [0.25, 0.5) → 1, [0.5, 0.75) → 2, [0.75, 1.0] → 3
    """
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"Probability must be between 0.0 and 1.0, got {prob}")
    
    # Edge case: prob = 1.0 should map to the highest class
    if prob == 1.0:
        return num_classes - 1
    
    # Map [0, 1) to class indices [0, num_classes-1]
    class_idx = int(prob * num_classes)
    return class_idx


class DifficultyClassificationDataset(Dataset):
    """
    Dataset for multi-class difficulty classification.
    Converts continuous probability labels into discrete difficulty classes.
    """
    def __init__(self, raw_data_list, tokenizer, num_classes=2, max_length=512):
        """
        Args:
            raw_data_list: List of dicts with 'question', 'correct_answers_num', 'total_answers_num'
            tokenizer: Tokenizer instance
            num_classes: Number of difficulty levels (default: 2 for easy/hard)
            max_length: Max sequence length
        """
        self.samples = []
        self.num_classes = num_classes
        
        for entry in raw_data_list:
            question = entry['question']
            correct_answers_num = entry['correct_answers_num']
            total_answers_num = entry['total_answers_num']
            
            # Compute empirical probability (correct rate)
            empirical_p = correct_answers_num / total_answers_num if total_answers_num > 0 else 0.0
            
            # Convert probability to difficulty class
            difficulty_class = probability_to_difficulty_class(empirical_p, num_classes)
            
            # Only use the question, no answer/solution
            text = f"Question: {question}"
            
            # Store both the class and the original probability for reference
            self.samples.append({
                "text": text,
                "label": difficulty_class,
                "prob": empirical_p  # Keep original for debugging/analysis
            })
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encodings = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False 
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(item["label"], dtype=torch.long)  # long for CE loss
        }