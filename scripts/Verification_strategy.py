from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal, Optional, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class BinaryVerifier:
    """
    Wrapper for  PEFT binary sequence-classification verifier.

    training code formats inputs like:
        "Question: ...\\nAnswer: ..."

    The model is an AutoModelForSequenceClassification with num_labels=1,
    so it returns a single raw **logit** per example.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 256

    def build_text(self, question: str, answer: str) -> str:
        return f"Question: {question}\nAnswer: {answer}"

    @torch.no_grad()
    def logits(self, question: str, answers: Sequence[str], batch_size: int = 8) -> list[float]:
        """
        Batched verifier scoring.

        Returns:
            List of raw logits (one per answer), aligned with `answers`.
        """
        if not answers:
            return []
        self.model.eval()
        device = next(self.model.parameters()).device

        out: list[float] = []
        for i in range(0, len(answers), batch_size):
            chunk = answers[i : i + batch_size]
            texts = [self.build_text(question, a) for a in chunk]
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = self.model(**enc)
            logits = outputs.logits.squeeze(-1)  # [B, 1] -> [B]
            # numpy doesn't support bfloat16; cast to float32 before moving to CPU
            out.extend([float(x) for x in logits.detach().float().cpu().tolist()])
        return out

    @torch.no_grad()
    def logit(self, question: str, answer: str) -> float:
        """Single-example convenience wrapper (still uses the model's raw logit)."""
        return self.logits(question, [answer], batch_size=1)[0]


def best_of_n(
    question: str,
    answers: Sequence[str],
    verifier: BinaryVerifier,
    *,
    batch_size: int = 8,
) -> str:
    """Return the candidate answer with the highest verifier logit."""
    if not answers:
        raise ValueError("answers must be non-empty")
    logits = verifier.logits(question, answers, batch_size=batch_size)
    best_idx = max(range(len(answers)), key=lambda i: logits[i])
    return answers[best_idx]


def rejection_sampling_from_candidates(
    *,
    question: str,
    answers: Sequence[str],
    verifier: BinaryVerifier,
    threshold: float,
) -> str:
    """
    Rejection sampling when you ALREADY have a list of candidate answers.

    Walk candidates in order, accept the first with sigmoid(logit) >= threshold.
    If none pass, return the best-seen (max logit).
    """
    if not answers:
        raise ValueError("answers must be non-empty")

    # Convert probability threshold -> logit threshold for a stable comparison.
    # sigmoid(logit) >= p  <=>  logit >= log(p/(1-p))
    p = float(threshold)
    p = min(max(p, 1e-6), 1 - 1e-6)
    logit_threshold = math.log(p / (1 - p))

    best_idx = 0
    best_logit = float("-inf")
    for i, ans in enumerate(answers):
        score = float(verifier.logit(question, ans))
        if score > best_logit:
            best_logit = score
            best_idx = i
        if score >= logit_threshold:
            return ans
    return answers[best_idx]


def rejection_sampling(
    *,
    question: str,
    generator: "AnswerGenerator",
    verifier: BinaryVerifier,
    threshold: float,
    max_trials: int,
    generation_kwargs: Optional[dict[str, Any]] = None,
) -> str:
    """
    Rejection sampling strategy (sequential generation):
    - generate ONE answer each trial
    - score it immediately with the verifier logit
    - accept first answer with sigmoid(logit) >= threshold
    - if we hit max_trials without acceptance, return the best-seen answer so far

    Note: If generation returns no answers, raises.
    """
    generation_kwargs = generation_kwargs or {}
    if max_trials <= 0:
        raise ValueError("max_trials must be > 0")

    seen_answers: list[str] = []
    seen_logits: list[float] = []

    p = float(threshold)
    p = min(max(p, 1e-6), 1 - 1e-6)
    logit_threshold = math.log(p / (1 - p))

    for _ in range(max_trials):
        ans = generator.generate_one(question, **generation_kwargs)
        if ans is None or ans == "":
            continue
        score = verifier.logit(question, ans)
        seen_answers.append(ans)
        seen_logits.append(float(score))
        if score >= logit_threshold:
            return ans

    # If we never accepted, return best-seen so far.
    if seen_answers:
        best_idx = max(range(len(seen_answers)), key=lambda i: seen_logits[i])
        return seen_answers[best_idx]

    raise ValueError("generator produced no usable answers during rejection sampling trials")


@dataclass
class AnswerGenerator:
    """
    Minimal wrapper around a text generation model. We intentionally keep this thin
    so you can plug in your own prompt template / decoding settings.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    build_prompt: Callable[[str], str]

    @torch.no_grad()
    def generate_one(
        self,
        question: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Optional[str]:
        device = next(self.model.parameters()).device
        prompt = self.build_prompt(question)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_ids = gen_ids[0][inputs["input_ids"].shape[1] :].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_n_batched(
        self,
        question: str,
        n: int,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> list[str]:
        """
        Generate N candidates in (typically) ONE call to `generate` using num_return_sequences.
        If the underlying model/tokenizer doesn't support it well, it still returns N strings.
        """
        if n <= 0:
            return []
        device = next(self.model.parameters()).device
        prompt = self.build_prompt(question)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        out: list[str] = []
        prompt_len = inputs["input_ids"].shape[1]
        for row in gen_ids:
            out_ids = row[prompt_len:].tolist()
            out.append(self.tokenizer.decode(out_ids, skip_special_tokens=True).strip())
        return out


def select_answer_with_generator(
    *,
    question: str,
    generator: AnswerGenerator,
    verifier: BinaryVerifier,
    strategy: Literal["best_of_n", "rejection_sampling"] = "best_of_n",
    n: int = 10,
    verifier_batch_size: int = 8,
    threshold: float = 0.5,
    generation_kwargs: Optional[dict[str, Any]] = None,
) -> str:
    """
    Top-level wrapper:
      - best_of_n: generate N answers in a batch, score in a batch, return argmax logit
      - rejection_sampling: generate 1 answer at a time, verify immediately, accept on threshold;
        if max trials reached, return best-seen; if no answers seen, fall back to best_of_n
      - returns the chosen answer
    """
    generation_kwargs = generation_kwargs or {}

    if strategy == "best_of_n":
        answers = generator.generate_n_batched(question, n, **generation_kwargs)
        if not answers:
            raise ValueError("generator produced no answers")
        return best_of_n(question, answers, verifier, batch_size=verifier_batch_size)

    # rejection_sampling (sequential generation)
    return rejection_sampling(
        question=question,
        generator=generator,
        verifier=verifier,
        threshold=threshold,
        max_trials=n,
        generation_kwargs=generation_kwargs,
    )


def select_answer_from_candidates(
    *,
    question: str,
    answers: Sequence[str],
    verifier: BinaryVerifier,
    strategy: Literal["best_of_n", "rejection_sampling"] = "best_of_n",
    verifier_batch_size: int = 8,
    threshold: float = 0.5,
) -> str:
    """
    Verify-only wrapper: you already have an `answers` list.
    """
    if strategy == "best_of_n":
        return best_of_n(question, answers, verifier, batch_size=verifier_batch_size)
    return rejection_sampling_from_candidates(
        question=question,
        answers=answers,
        verifier=verifier,
        threshold=threshold,
    )


def main():
    """
    Quick sanity-check runner:
    - loads the pAUC verifier checkpoint (state_dict) as the verifier model
    - evaluates on 10 questions from `data/verifier_dataset_train.json`
    - compares accuracy of:
        (1) always pick first answer
        (2) best-of-N using verifier logits
        (3) rejection sampling using verifier logits (threshold + fallback to best-so-far)

    Notes:
    - Uses verifier batch_size=1 to reduce GPU memory usage.
    - "Correct" means the chosen candidate's label is 1.

    How to run (from repo root):

    - pAUC verifier + test set (10 examples), rejection threshold = 0.7 (sigmoid prob):
        python Scripts/Verification_strategy.py --verifier pauc --n 10 --threshold 0.7

    - BCE verifier + test set (10 examples), rejection threshold = 0.5 (sigmoid prob):
        python Scripts/Verification_strategy.py --verifier bce --n 10 --threshold 0.5

    Args overview:
      --verifier {pauc,bce} : which checkpoint/config block to load
      --n        int         : number of questions to evaluate
      --threshold float      : rejection sampling threshold in sigmoid probability space (0~1)
    """
    import argparse
    import json
    import sys
    from pathlib import Path

    from transformers import AutoTokenizer

    # Make repo imports work no matter where we run from
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from Scripts.config_loader import load_config
    from Scripts.Lora_model import build_bce_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="How many questions to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Rejection sampling sigmoid(prob) threshold, e.g. 0.5")
    parser.add_argument(
        "--verifier",
        choices=["pauc", "bce"],
        default="pauc",
        help="Which verifier checkpoint to load (pauc uses PAUC_TRAIN.CHECKPOINT_PATH, bce uses BCE_TRAIN.CHECKPOINT_PATH)",
    )
    args = parser.parse_args()

    config = load_config("configure.yaml")

    # Load data (always from TEST set)
    data_path = repo_root / config.TEST_DATASET_PATH

    with open(data_path, "r", encoding="utf-8") as f:
        raw_questions = json.load(f)

    raw_questions = raw_questions[: args.n]
    if not raw_questions:
        raise ValueError("Dataset is empty.")

    # Pick verifier config + checkpoint
    if args.verifier == "pauc":
        vcfg = config.PAUC_TRAIN
        ckpt_path = repo_root / config.PAUC_TRAIN.CHECKPOINT_PATH
    else:
        vcfg = config.BCE_TRAIN
        ckpt_path = repo_root / config.BCE_TRAIN.CHECKPOINT_PATH

    # Build verifier model + load checkpoint (state_dict)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_bce_model(
        model_name=config.MODEL_NAME,
        lora_r=vcfg.LORA_R,
        lora_alpha=vcfg.LORA_ALPHA,
        lora_dropout=vcfg.LORA_DROPOUT,
        pad_token_id=tokenizer.pad_token_id,
        config=config,
    ).to(config.DEVICE)

    checkpoint = torch.load(ckpt_path, map_location=model.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    verifier = BinaryVerifier(model=model, tokenizer=tokenizer, max_length=vcfg.MAX_LENGTH)

    # Eval
    correct_first = 0
    correct_best = 0
    correct_reject = 0

    for ex in raw_questions:
        q = ex["question"]
        answers = list(ex["answers"])
        labels = list(ex["answer_labels"])

        ref = ex.get("reference_answer", None)
        if ref is not None:
            answers.append(ref)
            labels.append(1)

        # baseline: always pick the first answer
        correct_first += 1 if labels and int(labels[0]) == 1 else 0

        # best-of-N (verifier)
        chosen_best = select_answer_from_candidates(
            question=q,
            answers=answers,
            verifier=verifier,
            strategy="best_of_n",
            verifier_batch_size=1,  # per your request (low VRAM)
        )
        best_idx = answers.index(chosen_best)
        correct_best += 1 if int(labels[best_idx]) == 1 else 0

        # rejection sampling (verifier) over the provided candidate list
        chosen_rej = select_answer_from_candidates(
            question=q,
            answers=answers,
            verifier=verifier,
            strategy="rejection_sampling",
            verifier_batch_size=2,
            threshold=args.threshold,
        )
        rej_idx = answers.index(chosen_rej)
        correct_reject += 1 if int(labels[rej_idx]) == 1 else 0

    n = len(raw_questions)
    print("\n" + "=" * 60)
    print(f"Verifier type: {args.verifier}")
    print(f"Verifier checkpoint: {ckpt_path}")
    print(f"Dataset: {data_path} (n={n})")
    print(f"Device: {config.DEVICE}")
    print(f"Rejection threshold (sigmoid prob): {args.threshold}")
    print("-" * 60)
    print(f"Pick-first accuracy:     {correct_first}/{n} = {correct_first/n:.2%}")
    print(f"Best-of-N accuracy:      {correct_best}/{n} = {correct_best/n:.2%}")
    print(f"Rejection-sampling acc:  {correct_reject}/{n} = {correct_reject/n:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
