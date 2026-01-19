import os
import sys
import json
import torch
import torch.nn as nn
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from accelerate import Accelerator
from datetime import datetime

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.two_head_model import build_two_head_model
from scripts.verifier_dataset import Adaptive_N_VerifierDataset
from scripts.config_loader import load_config

def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def train_two_head():
    # 1. Setup Configuration & Accelerator
    config = load_config("config.yaml")
    hp = config.TWO_HEAD_TRAIN

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"two_head_{timestamp}"

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name="llm-verifier",
        config=vars(hp),
        # init_kwargs={"wandb": {"entity": "deeplearning-llm-verifier", "name": run_name}}

        init_kwargs={"wandb": {"name": run_name}}
    )

    device = accelerator.device

    # Paths
    checkpoint_path = hp.START_FROM_CHECKPOINT
    output_dir = os.path.join(hp.OUTPUT_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        print(f"Loading training data from {config.TRAIN_DATASET_PATH}...")

    with open(config.TRAIN_DATASET_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    if hp.DEBUG_SAMPLE_SIZE:
        raw_data = raw_data[:hp.DEBUG_SAMPLE_SIZE]

    random.seed(42)
    random.shuffle(raw_data)
    split_idx = int(0.9 * len(raw_data))
    train_raw = raw_data[:split_idx]
    val_raw = raw_data[split_idx:]

    train_dataset = Adaptive_N_VerifierDataset(train_raw, tokenizer, max_length=hp.MAX_LENGTH)
    val_dataset = Adaptive_N_VerifierDataset(val_raw, tokenizer, max_length=hp.MAX_LENGTH)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=hp.BATCH_SIZE, shuffle=False, collate_fn=collator)

    # 3. Build Model
    model = build_two_head_model(
        model_name=config.MODEL_NAME,
        device=device,
        checkpoint_path=checkpoint_path,
        lora_r=hp.LORA_R,
        lora_alpha=hp.LORA_ALPHA,
        lora_dropout=hp.LORA_DROPOUT,
        num_classes=None, # regression
        pad_token_id=tokenizer.pad_token_id,
        pooling="mean"
    )

    # Ensure head_b is trainable
    for param in model.head_b.parameters():
        param.requires_grad = True

    # 4. Optimizer & Loss
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=hp.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    num_training_steps = len(train_loader) * hp.EPOCH_NUM // hp.GRAD_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(hp.WARMUP_RATIO * num_training_steps),
        num_training_steps=num_training_steps
    )

    # Prepare for Accelerator
    accelerator.gradient_accumulation_steps = hp.GRAD_ACCUMULATION_STEPS
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if accelerator.is_main_process:
        print(f"\nStarting Stage 2 Training (Head B + Adapter B)")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    best_val_mae = float('inf')
    global_step = 0

    # 5. Training Loop
    for epoch in range(hp.EPOCH_NUM):
        model.train()
        total_loss = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process)

        for step, batch in progress_bar:
            with accelerator.accumulate(model):
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], head='b')
                logits = outputs.logits.squeeze(-1)

                loss = criterion(logits, batch['labels'])
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), hp.MAX_GRAD_NORM)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                current_loss = loss.item()
                total_loss += current_loss

                # Log Training to WandB
                accelerator.log({
                    "train/loss": current_loss,
                    "train/lr": scheduler.get_last_lr()[0]
                }, step=global_step)

                if (step + 1) % 50 == 0:
                    with torch.no_grad():
                        pred_prob = torch.sigmoid(logits)
                        mae = torch.abs(pred_prob - batch['labels']).mean().item()
                        accelerator.log({"train/mae": mae}, step=global_step)
                    progress_bar.set_postfix({'loss': f"{current_loss:.4f}", 'mae': f"{mae:.4f}"})

            # 6. Periodic Validation
            if global_step > 0 and global_step % hp.VAL_STEP_INTERVAL == 0:
                model.eval()
                val_mae = 0
                val_loss = 0

                if accelerator.is_main_process:
                    print(f"\nValidating at step {global_step}...")

                with torch.no_grad():
                    for batch in val_loader:
                        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], head='b')
                        logits = outputs.logits.squeeze(-1)

                        loss = criterion(logits, batch['labels'])

                        # Gather results from all GPUs
                        logits, labels = accelerator.gather_for_metrics((logits, batch['labels']))

                        val_loss += loss.item()
                        pred_prob = torch.sigmoid(logits)
                        val_mae += torch.abs(pred_prob - labels).sum().item()

                avg_val_mae = val_mae / len(val_dataset)
                avg_val_loss = val_loss / len(val_loader)

                # Log Validation to WandB
                accelerator.log({
                    "val/loss": avg_val_loss,
                    "val/mae": avg_val_mae,
                    "epoch": epoch + 1
                }, step=global_step)

                if accelerator.is_main_process:
                    print(f"Step {global_step} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f}")

                    if avg_val_mae < best_val_mae:
                        best_val_mae = avg_val_mae
                        print(f"New best model! Saving checkpoint...")

                        # Unwrap and save
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.bce_model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        ckpt_path = os.path.join(hp.CHECKPOINT_DIR, f"two_head_best.pt")
                        _ensure_parent_dir(ckpt_path)
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': unwrapped_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_mae': avg_val_mae,
                        }, ckpt_path)

                model.train() # Back to training mode

            global_step += 1

    accelerator.end_training()

if __name__ == "__main__":
    train_two_head()

if __name__ == "__main__":
    train_two_head()
