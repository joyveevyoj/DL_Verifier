import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from tqdm import tqdm
from Verifier_dataset import VerifierDataset
from Lora_model import build_bce_model
from transformers import (
    AutoTokenizer, 
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)
from libauc.metrics import auc_roc_score
import random


def _load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint.get('epoch', -1) + 1
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    print(f"Resuming BCE training from {path} (starting epoch {start_epoch}, best val {best_val_acc:.4f})")
    return start_epoch, best_val_acc


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
def train_bce(config, raw_questions):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if config.BCE_TRAIN.DEBUG_SAMPLE_SIZE:
        raw_questions = raw_questions[:config.BCE_TRAIN.DEBUG_SAMPLE_SIZE]
    
    random.seed(42)
    random.shuffle(raw_questions)
    split_idx = int(0.9 * len(raw_questions))
    if split_idx == 0 and len(raw_questions) > 0: 
        split_idx = 1
    train_questions = raw_questions[:split_idx]
    val_questions = raw_questions[split_idx:]

    train_dataset = VerifierDataset(train_questions, tokenizer, config.BCE_TRAIN.MAX_LENGTH)
    val_dataset = VerifierDataset(val_questions, tokenizer, config.BCE_TRAIN.MAX_LENGTH)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.BCE_TRAIN.BATCH_SIZE, shuffle=True, collate_fn=collator, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BCE_TRAIN.BATCH_SIZE, shuffle=False, collate_fn=collator, pin_memory=True)
    num_training_steps = len(train_loader) * config.BCE_TRAIN.EPOCH_NUM // config.BCE_TRAIN.GRAD_ACCUMULATION_STEPS
    
    # Build model with LoRA configuration
    model = build_bce_model(
        model_name=config.MODEL_NAME,
        lora_r=config.BCE_TRAIN.LORA_R,
        lora_alpha=config.BCE_TRAIN.LORA_ALPHA,
        lora_dropout=config.BCE_TRAIN.LORA_DROPOUT,
        pad_token_id=tokenizer.pad_token_id,
        config=config
    )
    model = model.to(config.DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=config.BCE_TRAIN.LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    best_val_acc = 0.0
    run_start_epoch = 0
    if config.BCE_TRAIN.START_FROM_CHECKPOINT:
        run_start_epoch, best_val_acc = _load_checkpoint(
            config.BCE_TRAIN.START_FROM_CHECKPOINT,
            model,
            optimizer,
            scheduler,
        )
    for epoch in range(run_start_epoch, config.BCE_TRAIN.EPOCH_NUM):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.BCE_TRAIN.EPOCH_NUM}")
        
        for step, batch in progress_bar:
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            
            logits = outputs.logits.squeeze(-1) 
            
            loss = criterion(logits, batch['labels'])
            loss = loss / config.BCE_TRAIN.GRAD_ACCUMULATION_STEPS
            
            loss.backward()
            
            if (step + 1) % config.BCE_TRAIN.GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * config.BCE_TRAIN.GRAD_ACCUMULATION_STEPS
            total_loss += current_loss
            progress_bar.set_postfix({'loss': current_loss})
            
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_correct = 0
        val_total = 0
        val_pred_list = []
        val_true_list = []
        
        print(f"Validating Epoch {epoch+1}...")
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
                
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = outputs.logits.squeeze(-1)

                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).float()
                # numpy doesn't support bfloat16; cast to float32 first
                val_pred_list.append(probs.detach().float().cpu().numpy())
                val_true_list.append(batch['labels'].cpu().numpy())
                val_correct += (predictions == batch['labels']).sum().item()
                val_total += len(batch['labels'])
        if len(val_true_list) > 0:
            val_pred = np.concatenate(val_pred_list)
            val_true = np.concatenate(val_true_list)
            val_pauc = auc_roc_score(val_true, val_pred, max_fpr=config.P_AUC_MAX_FPR)
        else:
            val_pauc = 0.0

        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1} Finished | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2%} | Val pAUC: {val_pauc:.4f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            print("New best model found. Saving checkpoint...")
            os.makedirs(config.BCE_TRAIN.OUTPUT_DIR, exist_ok=True)
            _ensure_parent_dir(config.BCE_TRAIN.CHECKPOINT_PATH)
            model.save_pretrained(config.BCE_TRAIN.OUTPUT_DIR)
            tokenizer.save_pretrained(config.BCE_TRAIN.OUTPUT_DIR)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, config.BCE_TRAIN.CHECKPOINT_PATH)
            print(f"Checkpoint Saved (best so far at epoch {epoch+1}).\n")
