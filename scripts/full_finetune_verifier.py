import os
import argparse
import json
import random
import numpy as np
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

# Assuming these are in your local project structure
from llm_verifier.verifier import MathAnswerVerifier
from llm_verifier.dataset import VerifierDataset


# use gpt2 as a demo because it's small and fast.
MODEL = "gpt2"


def set_global_seed(seed):
    """
    Fix the seed for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Helper from transformers
    set_seed(seed)
    print(f"[Info] Global seed set to: {seed}")


def evaluate(verifier, dataloader):
    """
    Evaluates the model on the validation set.
    Returns average loss and simple token accuracy.
    """
    model = verifier.model
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            # Calculate Accuracy
            logits = outputs.logits
            labels = batch["labels"]

            predictions = torch.argmax(logits, dim=-1)

            # Mask for valid labels (not -100)
            mask = labels != -100

            # Calculate correct predictions within the mask
            correct = (predictions[mask] == labels[mask]).sum().item()
            num_tokens = mask.sum().item()

            total_correct += correct
            total_tokens += num_tokens

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    return avg_loss, accuracy


def finetune_verifier(verifier, train_data, val_data=None, epochs=1, batch_size=4, lr=1e-5, warmup_ratio=0.1, eval_every=1, seed=42):
    """
    Finetune the verifier with a Learning Rate Scheduler and Validation Loop.
    """
    # 1. Prepare Training Data
    train_dataset = VerifierDataset(train_data, verifier)

    # Common collate function for padding
    def collate_fn(batch):
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [b["input_ids"] for b in batch],
                batch_first=True,
                padding_value=verifier.tokenizer.pad_token_id or verifier.tokenizer.eos_token_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [b["attention_mask"] for b in batch],
                batch_first=True,
                padding_value=0
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [b["labels"] for b in batch],
                batch_first=True,
                padding_value=-100
            ),
        }

    # Use a generator for DataLoader shuffling to be explicitly controlled by torch seed
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=g
    )

    # 2. Prepare Validation Data (if provided)
    val_dataloader = None
    if val_data:
        val_dataset = VerifierDataset(val_data, verifier)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        print(f"Training set size (flattened): {len(train_dataset)}")
        print(f"Validation set size (flattened): {len(val_dataset)}")
    else:
        print(f"Training set size (flattened): {len(train_dataset)}")

    # 3. Setup Optimizer
    model = verifier.model
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    print(f"Training for {total_steps} steps with {num_warmup_steps} warmup steps.")

    # 4. Training Loop
    for epoch in range(epochs):
        model.train() # Ensure model is in train mode at start of epoch
        total_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            current_lr = scheduler.get_last_lr()[0]
            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, step {step+1}, loss = {total_loss / (step+1):.4f}, lr = {current_lr:.2e}")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished, Training Loss = {avg_train_loss:.4f}")

        # 5. Validation Loop (Periodic)
        if val_dataloader and (epoch + 1) % eval_every == 0:
            print("="*80)
            print(f"Running Validation at end of Epoch {epoch+1}")
            print("="*80)
            val_loss, val_acc = evaluate(verifier, val_dataloader)
            print(f"Epoch {epoch+1} Validation Results -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            print("="*80)
            model.train() # Switch back to train mode

    # 6. Final Validation (if not already done by the loop)
    if val_dataloader and epochs % eval_every != 0:
        print("="*80)
        print(f"Running Final Validation")
        print("="*80)
        val_loss, val_acc = evaluate(verifier, val_dataloader)
        print(f"Final Validation Results -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    model.eval()
    return verifier


def main():
    parser = argparse.ArgumentParser(description="Finetune the verifier model.")
    # Modified line below:
    parser.add_argument("--data_path", type=str, default="data/verifier_dataset_train.json", help="Path to the training data file (JSONL format).")
    parser.add_argument("--val_split", type=float, default=0.1, help="Ratio of data to use for validation (0.0 to 1.0).")
    parser.add_argument("--eval_every", type=int, default=1, help="Run validation every m epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # ===============================================================
    # 0. Set Global Seed
    # ===============================================================
    set_global_seed(args.seed)

    # ===============================================================
    # 1. Setup Device
    # ===============================================================
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # ===============================================================
    # 2. Initialize Verifier
    # ===============================================================
    model_name = MODEL
    print(f"Loading model {model_name}...")
    verifier = MathAnswerVerifier(
        model_name=model_name,
        device=device,
        label_yes=" y",
        label_no=" n"
    )
    if verifier.tokenizer.pad_token is None:
        verifier.tokenizer.pad_token = verifier.tokenizer.eos_token
        verifier.model.config.pad_token_id = verifier.model.config.eos_token_id

    # ===============================================================
    # 3. Load and Split Data
    # ===============================================================
    print(f"Loading raw data from {args.data_path}...")

    # Check if file exists before opening to prevent crash
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at: {args.data_path}")
    try:
        with open(args.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            print("Successfully loaded raw data.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    demo_idx = random.randint(0, len(raw_data) - 1)
    demo_item = raw_data[demo_idx]
    print(f"\nData item example (index {demo_idx}):")
    print("Question:", demo_item["question"])
    print("Reference answer:", demo_item["reference_answer"][:100], "...")
    print("Number of candidate answers:", len(demo_item["answers"]))
    print("Number of labels:", len(demo_item["answer_labels"]))
    print("total_answers_num:", demo_item["total_answers_num"])
    print("correct_answers_num:", demo_item["correct_answers_num"])

    # Shuffle using the GLOBAL seed (random.seed was set in set_global_seed)
    random.shuffle(raw_data)

    # Calculate split index
    total_len = len(raw_data)
    val_len = int(total_len * args.val_split)

    # Perform Split
    val_data = raw_data[:val_len]
    train_data = raw_data[val_len:]

    print(f"\nTotal raw examples: {total_len}")
    print(f"Train split: {len(train_data)} questions")
    print(f"Val split:   {len(val_data)} questions")

    # ===============================================================
    # 4. Finetune Verifier
    # ===============================================================
    print("\n" + "="*80)
    print("START FINE-TUNING")
    print("="*80)

    verifier = finetune_verifier(
        verifier,
        train_data,
        val_data=val_data,
        epochs=3,
        batch_size=2,
        lr=5e-5,
        warmup_ratio=0.1,
        eval_every=args.eval_every,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
