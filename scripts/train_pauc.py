import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
from verifier_dataset import VerifierDataset
from lora_model import build_bce_model
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)
import random
import numpy as np
from libauc.losses import pAUCLoss
from libauc.optimizers import SOPAs
from libauc.sampler import DualSampler
from libauc.metrics import auc_roc_score
from datetime import datetime


def _load_checkpoint(path, model, optimizer, scheduler, accelerator):
    # Load to the correct device managed by accelerator
    checkpoint = torch.load(path, map_location=accelerator.device)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
    # Load optimizer and scheduler states
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint.get('epoch', -1) + 1
    start_global_step = checkpoint.get('global_step', -1) + 1
    best_val_pauc = checkpoint.get('best_val_pauc', 0.0)
    # Only print on the main process to avoid clutter
    if accelerator.is_local_main_process:
        print(f"Resuming training from {path} (starting epoch {start_epoch}, best val pauc {best_val_pauc:.4f})")
    return start_epoch, start_global_step, best_val_pauc


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _make_collator_with_index(tokenizer):
    """
    HF DataCollatorWithPadding may drop unknown keys (like 'index').
    This wrapper preserves it as a tensor batch['index'].
    """
    base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collate(features):
        indices = torch.tensor([int(f["index"]) for f in features], dtype=torch.long)
        features_wo_index = [{k: v for k, v in f.items() if k != "index"} for f in features]
        batch = base_collator(features_wo_index)
        batch["index"] = indices
        return batch

    return collate


def train_pAUC(config, raw_questions, accelerator, timestamp, mode="pauc"):
    accelerator.gradient_accumulation_steps = config.PAUC_TRAIN.GRAD_ACCUMULATION_STEPS

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.PAUC_TRAIN.DEBUG_SAMPLE_SIZE:
        raw_questions = raw_questions[:config.PAUC_TRAIN.DEBUG_SAMPLE_SIZE]

    random.seed(42)
    random.shuffle(raw_questions)
    split_idx = int(0.9 * len(raw_questions))
    if split_idx == 0 and len(raw_questions) > 0:
        split_idx = 1
    train_questions = raw_questions[:split_idx]
    val_questions = raw_questions[split_idx:]

    train_dataset = VerifierDataset(train_questions, tokenizer, config.PAUC_TRAIN.MAX_LENGTH)
    val_dataset = VerifierDataset(val_questions, tokenizer, config.PAUC_TRAIN.MAX_LENGTH)

    collator = _make_collator_with_index(tokenizer)
    # DualSampler ensures each batch has positives (1) and negatives (0).
    train_labels = [s["label"] for s in train_dataset.samples]
    sampler = DualSampler(train_dataset, batch_size=config.PAUC_TRAIN.BATCH_SIZE, labels=train_labels, sampling_rate=config.PAUC_TRAIN.SAMPLING_RATE)
    train_loader = DataLoader(train_dataset, batch_size=config.PAUC_TRAIN.BATCH_SIZE, sampler=sampler, shuffle=False, collate_fn=collator, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.PAUC_TRAIN.BATCH_SIZE, shuffle=False, collate_fn=collator, pin_memory=True)
    num_training_steps = len(train_loader) * config.PAUC_TRAIN.EPOCH_NUM // config.PAUC_TRAIN.GRAD_ACCUMULATION_STEPS
    num_warmup_steps = int(num_training_steps * config.PAUC_TRAIN.WARMUP_RATIO)

    # Build model with LoRA configuration
    model = build_bce_model(
        model_name=config.MODEL_NAME,
        lora_r=config.PAUC_TRAIN.LORA_R,
        lora_alpha=config.PAUC_TRAIN.LORA_ALPHA,
        lora_dropout=config.PAUC_TRAIN.LORA_DROPOUT,
        pad_token_id=tokenizer.pad_token_id,
        config=config
    )

    loss_fn = pAUCLoss('1w', data_len=len(train_dataset), margin=config.PAUC_TRAIN.MARGIN, gamma=config.PAUC_TRAIN.GAMMA)
    optimizer = SOPAs(model.parameters(), mode='adam', lr=config.PAUC_TRAIN.LEARNING_RATE, weight_decay=config.PAUC_TRAIN.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    run_start_epoch = 0
    best_val_pauc = 0.0
    global_step = 0

    if config.PAUC_TRAIN.START_FROM_CHECKPOINT:
        run_start_epoch, global_step, best_val_pauc = _load_checkpoint(
            config.PAUC_TRAIN.START_FROM_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            accelerator
        )

    for epoch in range(run_start_epoch, config.PAUC_TRAIN.EPOCH_NUM):
        total_loss = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process)

        for step, batch in progress_bar:
            model.train()
            with accelerator.accumulate(model):
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = outputs.logits.squeeze(-1)
                y_prob = torch.sigmoid(logits)

                loss = loss_fn(y_prob, batch['labels'], batch['index'])

                # Replaced backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Gradient clipping
                    accelerator.clip_grad_norm_(model.parameters(), config.PAUC_TRAIN.MAX_GRAD_NORM)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                current_loss = loss.item()
                total_loss += current_loss
                accelerator.log({"train/loss": current_loss, "train/lr": scheduler.get_last_lr()[0]}, step=global_step)
                progress_bar.set_postfix({'loss': current_loss})

            if global_step % config.PAUC_TRAIN.VAL_STEP_INTERVAL == 0:
                # Validation Loop
                model.eval()
                val_pred_list = []
                val_true_list = []

                if accelerator.is_local_main_process:
                    print(f"Validating Epoch {epoch+1}...")

                with torch.no_grad():
                    for batch in val_loader:
                        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                        logits = outputs.logits.squeeze(-1)
                        probs = torch.sigmoid(logits)

                        # Gather predictions and labels from all gpus
                        probs, labels = accelerator.gather_for_metrics((probs, batch['labels']))

                        val_pred_list.append(probs.detach().cpu().float().numpy())
                        val_true_list.append(labels.cpu().numpy())

                # Metrics calculation (only on main process)
                if accelerator.is_main_process:
                    if len(val_true_list) > 0:
                        val_pred = np.concatenate(val_pred_list)
                        val_true = np.concatenate(val_true_list)

                        predictions = (val_pred > 0.5).astype(float)
                        val_correct = (predictions == val_true).sum()
                        val_total = len(val_true)
                        val_acc = val_correct / val_total

                        try:
                            val_pauc = auc_roc_score(val_true, val_pred, max_fpr=config.P_AUC_MAX_FPR)
                        except:
                            val_pauc = 0.0
                    else:
                        val_acc = 0.0
                        val_pauc = 0.0

                    avg_train_loss = total_loss / (step + 1)
                    print(f"Epoch {epoch+1} Progress | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2%} | Val pAUC: {val_pauc:.4f}")

                    # Log Validation Metrics
                    accelerator.log({
                        "val/acc": val_acc,
                        "val/pauc": val_pauc,
                        "epoch": epoch + 1
                    }, step=global_step)

                    if val_pauc >= best_val_pauc:
                        best_val_pauc = val_pauc
                        print(f"New best model found (pAUC: {best_val_pauc:.4f}). Saving checkpoint...")
                        outputs_dir = os.path.join(config.PAUC_TRAIN.OUTPUT_DIR, f"{mode}_{timestamp}")
                        os.makedirs(outputs_dir, exist_ok=True)

                        checkpoint_path = os.path.join(config.PAUC_TRAIN.CHECKPOINT_DIR, f"{mode}_{timestamp}", f"best_model_step_{global_step}.pt")
                        _ensure_parent_dir(checkpoint_path)

                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(outputs_dir)
                        tokenizer.save_pretrained(outputs_dir)
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': unwrapped_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_val_pauc': best_val_pauc,
                        }, checkpoint_path)
                        print(f"Checkpoint Saved to {checkpoint_path} (best so far at step {global_step}, epoch {epoch+1}).\n")

            global_step += 1

    accelerator.end_training()
