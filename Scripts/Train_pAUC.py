import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from Verifier_dataset import VerifierDataset
from Lora_model import build_bce_model
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


def _load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint.get('epoch', -1) + 1
    best_val = checkpoint.get('best_val_acc', 0.0)
    print(f"Resuming pAUC training from {path} (starting epoch {start_epoch}, best val {best_val:.4f})")
    return start_epoch, best_val


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

def train_pAUC(config, raw_questions):
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
    sampler = DualSampler(train_dataset, batch_size=config.PAUC_TRAIN.BATCH_SIZE, labels=train_labels, sampling_rate=0.5)
    train_loader = DataLoader(train_dataset, batch_size=config.PAUC_TRAIN.BATCH_SIZE, sampler=sampler, shuffle=False, collate_fn=collator, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.PAUC_TRAIN.BATCH_SIZE, shuffle=False, collate_fn=collator, pin_memory=True)
    num_training_steps = len(train_loader) * config.PAUC_TRAIN.EPOCH_NUM // config.PAUC_TRAIN.GRAD_ACCUMULATION_STEPS
    
    # Build model with LoRA configuration
    model = build_bce_model(
        model_name=config.MODEL_NAME,
        lora_r=config.PAUC_TRAIN.LORA_R,
        lora_alpha=config.PAUC_TRAIN.LORA_ALPHA,
        lora_dropout=config.PAUC_TRAIN.LORA_DROPOUT,
        pad_token_id=tokenizer.pad_token_id,
        config=config
    )
    model = model.to(config.DEVICE)
    loss_fn = pAUCLoss('1w', data_len=len(train_dataset), margin=config.PAUC_TRAIN.MARGIN, gamma=config.PAUC_TRAIN.GAMMA)
    optimizer = SOPAs(model.parameters(), mode='adam', lr=config.PAUC_TRAIN.LEARNING_RATE, weight_decay=config.PAUC_TRAIN.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    run_start_epoch = 0
    best_val_acc = 0.0
    if config.PAUC_TRAIN.START_FROM_CHECKPOINT:
        run_start_epoch, best_val_acc = _load_checkpoint(
            config.PAUC_TRAIN.START_FROM_CHECKPOINT,
            model,
            optimizer,
            scheduler,
        )
    for epoch in range(run_start_epoch, config.PAUC_TRAIN.EPOCH_NUM):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.PAUC_TRAIN.EPOCH_NUM}")
        
        for step, batch in progress_bar:
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            
            logits = outputs.logits.squeeze(-1) 
            y_prob = torch.sigmoid(logits)
            loss = loss_fn(y_prob, batch['labels'], batch['index'])
            loss = loss / config.PAUC_TRAIN.GRAD_ACCUMULATION_STEPS
            
            loss.backward()
            
            if (step + 1) % config.PAUC_TRAIN.GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * config.PAUC_TRAIN.GRAD_ACCUMULATION_STEPS
            total_loss += current_loss
            progress_bar.set_postfix({'loss': current_loss})
            
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_pred_list = []
        val_true_list = []
        
        print(f"Validating Epoch {epoch+1}...")
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = outputs.logits.squeeze(-1)
                y_prob = torch.sigmoid(logits)
                # numpy doesn't support bfloat16; cast to float32 first
                val_pred_list.append(y_prob.detach().float().cpu().numpy())
                val_true_list.append(batch['labels'].cpu().numpy())

        val_pred = np.concatenate(val_pred_list)
        val_true = np.concatenate(val_true_list)
        val_pauc = auc_roc_score(val_true, val_pred, max_fpr=config.P_AUC_MAX_FPR)
        print(f"Epoch {epoch+1} Finished | Train Loss: {avg_train_loss:.4f} | Val pAUC: {val_pauc:.4f}")
        if val_pauc >= best_val_acc:
            best_val_acc = val_pauc
            print("New best model found. Saving checkpoint...")
            os.makedirs(config.PAUC_TRAIN.OUTPUT_DIR, exist_ok=True)
            _ensure_parent_dir(config.PAUC_TRAIN.CHECKPOINT_PATH)
            model.save_pretrained(config.PAUC_TRAIN.OUTPUT_DIR)
            tokenizer.save_pretrained(config.PAUC_TRAIN.OUTPUT_DIR)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, config.PAUC_TRAIN.CHECKPOINT_PATH)
            print(f"Checkpoint Saved (best so far at epoch {epoch+1}).\n")
