from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal, Optional, Sequence, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import sys
from pathlib import Path

# Add project root to sys.path to allow 'from Scripts...' imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Scripts.two_head_model import TwoHeadModel, build_two_head_model, compute_adaptive_n
from Scripts.config_loader import load_config
from Scripts.lora_model import build_bce_model


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
    verifier: Union[BinaryVerifier, TwoHeadVerifier],
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
    question: str,
    answers: Sequence[str],
    verifier: Union[BinaryVerifier, TwoHeadVerifier],
    threshold: float,
) -> tuple[str, int]:
    """
    Rejection sampling when you ALREADY have a list of candidate answers.

    Walk candidates in order, accept the first with sigmoid(logit) >= threshold.
    If none pass, return the best-seen (max logit).

    Returns:
        (chosen_answer, samples_used)
    """
    if not answers:
        raise ValueError("answers must be non-empty")

    # Convert probability threshold -> logit threshold for a stable comparison.
    p = float(threshold)
    p = min(max(p, 1e-6), 1 - 1e-6)
    logit_threshold = math.log(p / (1 - p))

    best_idx = 0
    best_logit = float("-inf")
    samples_used = 0
    for i, ans in enumerate(answers):
        samples_used += 1
        score = float(verifier.logit(question, ans))
        if score > best_logit:
            best_logit = score
            best_idx = i
        if score >= logit_threshold:
            return ans, samples_used
    return answers[best_idx], samples_used


def rejection_sampling(
    question: str,
    generator: "AnswerGenerator",
    verifier: Union[BinaryVerifier, TwoHeadVerifier],
    threshold: float,
    max_trials: int,
    generation_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[str, int]:
    """
    # Rejection sampling strategy (sequential generation):
    # - generate ONE answer each trial
    # - score it immediately with the verifier logit
    # - accept first answer with sigmoid(logit) >= threshold
    # - if we hit max_trials without acceptance, return the best-seen answer so far

    Returns:
        (chosen_answer, samples_used)
    """
    generation_kwargs = generation_kwargs or {}
    if max_trials <= 0:
        raise ValueError("max_trials must be > 0")

    seen_answers: list[str] = []
    seen_logits: list[float] = []

    p = float(threshold)
    p = min(max(p, 1e-6), 1 - 1e-6)
    logit_threshold = math.log(p / (1 - p))

    samples_used = 0
    for _ in range(max_trials):
        samples_used += 1
        ans = generator.generate_one(question, **generation_kwargs)
        if ans is None or ans == "":
            continue
        score = verifier.logit(question, ans)
        seen_answers.append(ans)
        seen_logits.append(float(score))
        if score >= logit_threshold:
            return ans, samples_used

    # If we never accepted, return best-seen so far.
    if seen_answers:
        best_idx = max(range(len(seen_answers)), key=lambda i: seen_logits[i])
        return seen_answers[best_idx], samples_used

    raise ValueError("generator produced no usable answers during rejection sampling trials")


@dataclass
class TwoHeadVerifier:
    """
    Wrapper for the TwoHeadModel.
    - Head A: Verifier (is this answer correct?)
    - Head B: Difficulty Estimator (overall correct rate for this question)
    """
    model: TwoHeadModel
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 256

    @torch.no_grad()
    def predict_correct_rate(self, question: str) -> float:
        """Uses Head B to predict question difficulty."""
        self.model.eval()
        device = next(self.model.parameters()).device
        text = f"Question: {question}"
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(device)
        out = self.model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"), head="b")
        prob = torch.sigmoid(out.logits).squeeze().cpu().item()
        return float(prob)

    @torch.no_grad()
    def logits(self, question: str, answers: Sequence[str], batch_size: int = 8) -> list[float]:
        """Uses Head A to score candidate answers."""
        if not answers:
            return []
        self.model.eval()
        device = self.model.device
        out: list[float] = []
        for i in range(0, len(answers), batch_size):
            chunk = answers[i : i + batch_size]
            texts = [f"Question: {question}\nAnswer: {a}" for a in chunk]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(device)
            res = self.model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"), head="a")
            logits = res.logits.squeeze(-1).float().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            out.extend(logits)
        return out

    @torch.no_grad()
    def logit(self, question: str, answer: str) -> float:
        """Single-example convenience wrapper."""
        return self.logits(question, [answer], batch_size=1)[0]


def adaptive_n(
    question: str,
    verifier: TwoHeadVerifier,
    generator: Optional[AnswerGenerator] = None,
    answers: Optional[Sequence[str]] = None,
    lambd: float = 0.01,
    max_batch_size: int = 8,
) -> tuple[str, int]:
    """
    Adaptive N Strategy:
    1. Predict N* based on question difficulty (Head B).
    2. Obtain N* candidate answers (via generator or sampling from input list).
    3. Choose best based on verifier scores (Head A).

    Returns:
        (chosen_answer, n_star)
    """
    # 1. Predict optimal N
    p_hat = verifier.predict_correct_rate(question)
    n_star = int(compute_adaptive_n(p_hat, lambd))

    # 2. Get candidates
    candidates: list[str] = []
    if generator is not None:
        # Dynamic Mode: Generate N* answers
        candidates = generator.generate_one(question, n=n_star)  # type: ignore
    elif answers is not None:
        # Static Mode: Sample N* from provided list (or take all if fewer)
        import random

        # Sample n_star items.
        n_to_take = min(n_star, len(answers))
        candidates = random.sample(list(answers), n_to_take)
    else:
        raise ValueError("Either 'generator' or 'answers' must be provided for adaptive_n.")

    if not candidates:
        return "", n_star

    # 3. Score and select best
    logits = verifier.logits(question, candidates, batch_size=max_batch_size)
    best_idx = int(torch.tensor(logits).argmax().item())
    return candidates[best_idx], n_star


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
    verifier: Union[BinaryVerifier, TwoHeadVerifier],
    strategy: Literal["best_of_n", "rejection_sampling", "adaptive_n"] = "best_of_n",
    verifier_batch_size: int = 8,
    threshold: float = 0.5,
    lambd: float = 0.01,
) -> str:
    """
    Verify-only wrapper: you already have an `answers` list.
    """
    if strategy == "best_of_n":
        return best_of_n(question, answers, verifier, batch_size=verifier_batch_size)
    if strategy == "rejection_sampling":
        return rejection_sampling_from_candidates(
            question=question,
            answers=answers,
            verifier=verifier,
            threshold=threshold,
        )
    if strategy == "adaptive_n":
        if not isinstance(verifier, TwoHeadVerifier):
            raise TypeError("adaptive_n strategy requires a TwoHeadVerifier.")
        return adaptive_n(
            question=question,
            verifier=verifier,
            answers=answers,
            lambd=lambd,
            max_batch_size=verifier_batch_size
        )
    raise ValueError(f"Unknown strategy: {strategy}")


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="How many questions to evaluate")
    parser.add_argument("--best_n", type=int, default=32, help="N for the Best-of-N strategy")
    parser.add_argument("--threshold", type=float, default=0.5, help="Rejection sampling sigmoid(prob) threshold, e.g. 0.5")
    parser.add_argument("--lambd", type=float, default=0.01, help="Cost coefficient for adaptive_n")
    parser.add_argument(
        "--strategy",
        choices=["best_of_n", "rejection_sampling", "adaptive_n"],
        default="best_of_n",
        help="Which strategy to use",
    )
    parser.add_argument(
        "--verifier",
        choices=["pauc", "bce", "two_head"],
        default="pauc",
        help="Which verifier checkpoint to load",
    )
    args = parser.parse_args()

    config = load_config("config.yaml")

    # Load data (always from TEST set)
    data_path = repo_root / config.TEST_DATASET_PATH

    with open(data_path, "r", encoding="utf-8") as f:
        raw_questions = json.load(f)

    raw_questions = raw_questions[: args.n]
    if not raw_questions:
        raise ValueError("Dataset is empty.")

    # Build verifier model + load checkpoint (state_dict)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pick verifier config + checkpoint
    if args.verifier == "pauc":
        vcfg = config.PAUC_TRAIN
        ckpt_path = repo_root / config.PAUC_TRAIN.CHECKPOINT_PATH
        # Build verifier model + load checkpoint (state_dict)
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
    elif args.verifier == "bce":
        vcfg = config.BCE_TRAIN
        ckpt_path = repo_root / config.BCE_TRAIN.CHECKPOINT_PATH
        # Build verifier model + load checkpoint (state_dict)
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
    else:  # two_head
        vcfg = config.TWO_HEAD_TRAIN
        ckpt_path = repo_root / config.TWO_HEAD_TRAIN.CHECKPOINT_DIR / "two_head_best_pauc.pt"
        model = build_two_head_model(
            model_name=config.MODEL_NAME,
            device=config.DEVICE,
            checkpoint_path=vcfg.START_FROM_CHECKPOINT,
            lora_r=vcfg.LORA_R,
            lora_alpha=vcfg.LORA_ALPHA,
            lora_dropout=vcfg.LORA_DROPOUT,
            num_classes=None,
            pad_token_id=tokenizer.pad_token_id,
            pooling="mean",
            config=config,
        )
        if ckpt_path.exists():
            print(f"Loading Stage 2 TwoHead weights from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=model.device)
            model.load_state_dict(checkpoint["model_state_dict"])
        verifier = TwoHeadVerifier(model=model, tokenizer=tokenizer, max_length=vcfg.MAX_LENGTH)

    model.eval()

    # Eval
    correct_random = 0
    correct_best = 0
    correct_reject = 0
    correct_adaptive = 0
    total_n_star = 0
    total_rej_samples = 0
    total_q = len(raw_questions)

    import random
    random.seed(42)  # For reproducibility

    print(f"\nEvaluating {total_q} questions...")
    for idx, ex in enumerate(raw_questions):
        q = ex["question"]
        answers = list(ex["answers"])
        labels = list(ex["answer_labels"])
        # ref = ex.get("reference_answer", None)
        # if ref is not None:
        #     answers.append(ref)
        #     labels.append(1)

        if not answers:
            continue

        # Shuffle candidates 
        combined = list(zip(answers, labels))
        random.shuffle(combined)
        answers, labels = zip(*combined)
        answers, labels = list(answers), list(labels)

        # Slice answers for Best-of-N and Rejection Sampling based on --best_n
        # (Adaptive N handles its own slicing/sampling)
        comp_answers_best = answers[: args.best_n]
        comp_labels_best = labels[: args.best_n]

        if not comp_answers_best:
            continue

        # 1. Random Sampling (Baseline)
        correct_random += 1 if int(comp_labels_best[0]) == 1 else 0

        # 2. Best-of-N (Head A)
        chosen_best = best_of_n(q, comp_answers_best, verifier, batch_size=1)
        correct_best += 1 if int(comp_labels_best[comp_answers_best.index(chosen_best)]) == 1 else 0

        # 3. Rejection Sampling (Head A)
        # For rejection sampling, use a default budget of 32 even if --best_n is different
        comp_answers_rej = answers[:32]
        comp_labels_rej = labels[:32]
        chosen_rej, rej_samples = rejection_sampling_from_candidates(q, comp_answers_rej, verifier, threshold=args.threshold)
        correct_reject += 1 if int(comp_labels_rej[comp_answers_rej.index(chosen_rej)]) == 1 else 0
        total_rej_samples += rej_samples

        # 4. Adaptive N (Head B + Head A) - Only for TwoHead
        # Use full answers list for adaptive sampling to respect its budget
        n_star_str = ""
        if args.verifier == "two_head":
            chosen_adapt, n_star = adaptive_n(q, verifier, answers=answers, lambd=args.lambd)
            correct_adaptive += 1 if int(labels[answers.index(chosen_adapt)]) == 1 else 0
            total_n_star += n_star
            n_star_str = f" | Adaptive N*={n_star}"

        if (idx + 1) % 1 == 0:
            print(f"[{idx+1}/{total_q}] Processing... Rej Samples={rej_samples}{n_star_str}")

    print("\n" + "=" * 60)
    print("VERIFICATION STRATEGY COMPARISON")
    print("=" * 60)
    print(f"Verifier Type: {args.verifier}")
    print(f"Checkpoint:    {ckpt_path}")
    print(f"Dataset:       {config.TEST_DATASET_PATH} (n={total_q})")
    print(f"Fixed N (BoN): {args.best_n}")
    print(f"Avg Rej Samples: {total_rej_samples/total_q:.2f}")
    if args.verifier == "two_head":
        print(f"Lambda:        {args.lambd}")
        print(f"Average N*:    {total_n_star/total_q:.2f}")
    print("-" * 60)
    print(f"Random Sampling (Pick First): {correct_random/total_q:7.2%} ({correct_random}/{total_q})")
    print(f"Best-of-N (Head A):           {correct_best/total_q:7.2%} ({correct_best}/{total_q})")
    print(f"Rejection Sampling (Head A):  {correct_reject/total_q:7.2%} ({correct_reject}/{total_q})")
    if args.verifier == "two_head":
        print(f"Adaptive N (Head B + A):      {correct_adaptive/total_q:7.2%} ({correct_adaptive}/{total_q})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
