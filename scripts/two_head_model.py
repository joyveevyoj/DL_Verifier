from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, Optional, Union

import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel

import sys
from pathlib import Path

# Add project root to sys.path to allow 'from scripts...' imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.lora_model import build_bce_model


def compute_adaptive_n(p_hat: Union[float, torch.Tensor], lambd: float) -> Union[int, torch.Tensor]:
    """
    Computes the optimal sampling budget N* by maximizing the objective:
    [1 - (1 - p_hat)^N - lambda * N]

    Formula: N* = round( log( lambda / -log(1 - p_hat) ) / log(1 - p_hat) )

    Args:
        p_hat: Predicted probability that a single sample is correct (Head B output).
        lambd: Cost coefficient for additional samples.

    Returns:
        Optimal integer budget N* (clamped to minimum of 1).
    """
    if isinstance(p_hat, torch.Tensor):
        # Prevent log domain errors
        p = p_hat.clamp(1e-6, 1 - 1e-6)
        log_1_p = torch.log(1 - p)
        # N* calculation
        numerator = torch.log(lambd / -log_1_p)
        n_star = numerator / log_1_p
        return torch.round(n_star).clamp(min=1).long()
    else:
        # Scalar float implementation
        p = max(1e-6, min(1 - 1e-6, p_hat))
        log_1_p = math.log(1 - p)
        try:
            numerator = math.log(lambd / -log_1_p)
            n_star = numerator / log_1_p
            return max(1, int(round(n_star)))
        except (ValueError, ZeroDivisionError):
            return 1


@dataclass
class TwoHeadOutput:
    """
    Output container for the two-head verifier model.

    - logits: head A output (binary), shape [batch, 1]
    - logits_b: head B output (regression or classification), shape:
        - [batch, 1] if num_classes is None (probability regression)
        - [batch, num_classes] if num_classes is set
    """

    logits: torch.Tensor
    logits_b: torch.Tensor


def _get_transformer_from_bce_model(bce_model: PreTrainedModel) -> nn.Module:
    """
    Extract the underlying transformer (the part that produces last_hidden_state)
    from a PEFT-wrapped AutoModelForSequenceClassification.
    """
    # PEFT model typically: bce_model.base_model.model -> underlying HF model
    base = getattr(bce_model, "base_model", None)
    if base is not None:
        inner = getattr(base, "model", None)
        if inner is not None:
            # For Qwen3ForSequenceClassification, the transformer is usually `.model`
            transformer = getattr(inner, "model", None)
            if transformer is not None:
                return transformer
            return inner

    # Non-PEFT fallback: HF model might have `.model` as transformer
    transformer = getattr(bce_model, "model", None)
    if transformer is not None:
        return transformer

    raise AttributeError("Unable to locate base transformer module inside bce_model.")


class TwoHeadModel(nn.Module):
    """
    Two-head model on top of a base transformer encoder/decoder output.

    This follows the pattern from `notebooks/adaptive_N_finetune.ipynb`:
      - base_model = AutoModel (no lm_head / no classifier head)
      - apply LoRA to attention projections
      - head_a: Linear(hidden_size -> 1) (binary)
      - head_b: 2-layer MLP (hidden_size -> hidden_size -> output_dim)
      - pooling: mean pooling over sequence (masked), optional last-token pooling
    """

    def __init__(
        self,
        bce_model: PreTrainedModel,
        head_b: nn.Module,
        *,
        pooling: Literal["mean", "last"] = "mean",
    ):
        super().__init__()
        self.bce_model = bce_model
        self.transformer = _get_transformer_from_bce_model(bce_model)
        self.head_b = head_b
        self.pooling = pooling

    @property
    def device(self) -> torch.device:
        """Helper to get the model device."""
        return next(self.parameters()).device

    def _pool(self, last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # last_hidden: [batch, seq_len, hidden]
        if self.pooling == "last":
            if attention_mask is None:
                return last_hidden[:, -1, :]
            seq_lens = attention_mask.sum(dim=1) - 1  # index of last non-padding token
            seq_lens = seq_lens.clamp(min=0)
            return last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lens]

        # default: mean pooling with masking
        if attention_mask is None:
            return last_hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        masked_hidden = last_hidden * mask
        summed = masked_hidden.sum(dim=1)  # [batch, hidden]
        counts = mask.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
        return summed / counts

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, head: str = "both"):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            head: 'a' (Q+A verifier), 'b' (Q-only difficulty), or 'both'

        Returns:
            - if head == 'a': SimpleNamespace(logits=logits_a)
            - if head == 'b': SimpleNamespace(logits=logits_b)
            - if head == 'both': TwoHeadOutput(logits=logits_a, logits_b=logits_b)
        """
        is_peft = hasattr(self.bce_model, "set_adapter")

        logits_a = None
        if head in ("a", "both"):
            # Head A: binary verifier (expects Q+A)
            if is_peft:
                # If we have a 'difficulty' adapter active, disable it for Head A
                # so it uses the merged/base weights only.
                with self.bce_model.disable_adapter():
                    logits_a = self.bce_model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                logits_a = self.bce_model(input_ids=input_ids, attention_mask=attention_mask).logits

        logits_b = None
        if head in ("b", "both"):
            # Head B: difficulty regression (expects Q only)
            if is_peft:
                self.bce_model.set_adapter("difficulty")

            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            pooled = self._pool(last_hidden, attention_mask)
            # Match dtype to head weights
            target_dtype = self.head_b[0].weight.dtype if isinstance(self.head_b, nn.Sequential) else torch.float32
            pooled = pooled.to(target_dtype)
            logits_b = self.head_b(pooled)

        if head == "a":
            return SimpleNamespace(logits=logits_a)
        if head == "b":
            return SimpleNamespace(logits=logits_b)
        return TwoHeadOutput(logits=logits_a, logits_b=logits_b)


def build_two_head_model(
    model_name: str,
    *,
    device: Union[str, torch.device] = "cpu",
    checkpoint_path: Optional[str] = None,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    num_classes: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    pooling: Literal["mean", "last"] = "mean",
    dtype: torch.dtype = torch.bfloat16,
    config: Optional[object] = None,
) -> TwoHeadModel:
    """
    Build a two-head model.
    If checkpoint_path is provided, it follows the 'Merged Path':
      1. Loads Stage 1 weights (Head A + LoRA A).
      2. Merges LoRA A into the base model weights.
      3. Adds a fresh LoRA adapter ('difficulty') for Stage 2 (Head B).
    """
    # 1. Build initial Stage 1 model
    bce_model = build_bce_model(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        pad_token_id=pad_token_id,
        config=config,
    ).to(device)

    # 2. If checkpoint provided, merge Stage 1 knowledge
    if checkpoint_path:
        print(f"Loading Stage 1 weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        bce_model.load_state_dict(state_dict, strict=False)

        print("Merging Stage 1 LoRA into base weights (Sequential Path)...")
        bce_model = bce_model.merge_and_unload()

        # 3. Add fresh LoRA for Stage 2
        print("Adding Stage 2 LoRA adapter ('difficulty')...")
        from peft import LoraConfig, get_peft_model
        config_b = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            # Use task_type=None to keep it flexible for our custom heads
        )
        bce_model = get_peft_model(bce_model, config_b, adapter_name="difficulty")

    transformer = _get_transformer_from_bce_model(bce_model)
    hidden_size = transformer.config.hidden_size
    output_dim = 1 if num_classes is None else int(num_classes)

    head_b = nn.Sequential(
        nn.Linear(hidden_size, hidden_size, bias=True),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, output_dim, bias=True),
    ).to(device=device, dtype=dtype)

    model = TwoHeadModel(bce_model=bce_model, head_b=head_b, pooling=pooling)
    return model.to(device)


def main():
    """
    Simple smoke test:
    - Load `checkpoints/bce_best_model.pt` into Head A (the BCE verifier part)
    - Sample a few questions from `data/verifier_dataset_test.json`
    - Run a forward pass for Head A + Head B and print:
        - true T/F label for each candidate
        - Head A prob = sigmoid(logit_a)
        - Head B predicted correct rate = sigmoid(logit_b)  (Head B is randomly initialized unless trained)
        - question-level true correct rate = (#label==1) / (#candidates)

    Run (repo root):
        python scripts/two_head_model.py --checkpoint checkpoints/bce_best_model.pt --n_questions 5 --k_answers 3
    """
    import argparse
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(repo_root / "checkpoints" / "bce_best_model.pt"))
    parser.add_argument("--dataset", type=str, default=str(repo_root / "data" / "verifier_dataset_test.json"))
    parser.add_argument("--n_questions", type=int, default=5)
    parser.add_argument("--k_answers", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    two_head = build_two_head_model(
        model_name=model_name,
        device=device,
        checkpoint_path=args.checkpoint,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        num_classes=None,  # regression head (1 dim)
        pad_token_id=tokenizer.pad_token_id,
        pooling=args.pooling,
        config=None,
    )
    two_head.eval()

    print("\n[Two-Stage Model Built]")
    print(f"  Stage 1 (Head A): Merged from {args.checkpoint}")
    print("  Stage 2 (Head B): Active via 'difficulty' adapter")

    # Load dataset
    with open(args.dataset, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw = raw[: args.n_questions]
    if not raw:
        raise ValueError("Dataset is empty.")

    print("\n" + "=" * 80)
    print(f"Device: {device} | dataset: {args.dataset} | n_questions={len(raw)} | k_answers={args.k_answers}")
    print("NOTE: Head B is randomly initialized unless you trained/saved it separately.")
    print("=" * 80)

    for qi, ex in enumerate(raw):
        q = ex["question"]
        answers = list(ex.get("answers", []))
        labels = list(ex.get("answer_labels", []))
        ref = ex.get("reference_answer", None)
        if ref is not None:
            answers.append(ref)
            labels.append(1)

        if not answers:
            continue

        true_rate = float(sum(labels)) / float(len(labels)) if labels else 0.0
        print(f"\n[Q{qi+1}] true_correct_rate={true_rate:.3f}")
        print(f"Question: {q[:140]}{'...' if len(q) > 140 else ''}")

        # 1. Compute Head B (Question only)
        q_enc = tokenizer(f"Question: {q}", return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out_b = two_head(input_ids=q_enc["input_ids"], attention_mask=q_enc.get("attention_mask"), head="b")
            prob_b = float(torch.sigmoid(out_b.logits).squeeze().cpu().item())

        # Compute N* with a few sample lambda values
        n_star_01 = compute_adaptive_n(prob_b, lambd=0.01)
        n_star_05 = compute_adaptive_n(prob_b, lambd=0.05)

        print(f"  -> HeadB Predicted Correct Rate: {prob_b:.4f}")
        print(f"  -> Adaptive N*: lambda=0.01 -> N={n_star_01} | lambda=0.05 -> N={n_star_05}")

        # 2. Compute Head A (Question + Answer) for each candidate
        for ai, (ans, y) in enumerate(list(zip(answers, labels))[: args.k_answers]):
            text = f"Question: {q}\nAnswer: {ans}"
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out_a = two_head(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"), head="a")
                prob_a = float(torch.sigmoid(out_a.logits).squeeze().cpu().item())

            print(f"  [A{ai+1}] true={str(bool(y)):5} | pred={str(prob_a > 0.5):5} | headA_verifier_prob={prob_a:.4f}")


if __name__ == "__main__":
    main()
