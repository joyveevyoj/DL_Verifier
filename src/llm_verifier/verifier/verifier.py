import math

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


class MathAnswerVerifier:
    def __init__(self, model_name: str, device: torch.device,
                 label_yes: str = " y", label_no: str = " n"):
        """
        A simple verifier built on top of a decoder-only LLM
        (e.g., Qwen / GPT-2 style AutoModelForCausalLM).

        Given (question, answer), it estimates:
            P(correct | question, answer) in (0, 1)

        Args:
            model_name: HuggingFace model name, e.g. "Qwen/Qwen2.5-0.5B" or "gpt2".
            device: "cuda" / "cpu".
            label_yes: text label representing "correct" (here: ' y').
            label_no: text label representing "incorrect" (here: ' n').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
        )

        if device is not None:
            self.model.to(device)

        # Store labels (for training you will reuse them)
        self.label_yes = label_yes
        self.label_no = label_no

        # Pre-tokenize yes/no label sequences
        # IMPORTANT: We tokenize them separately to append them later by ID,
        # avoiding string concatenation artifacts.
        self.yes_ids = self.tokenizer(label_yes, add_special_tokens=False).input_ids
        self.no_ids = self.tokenizer(label_no, add_special_tokens=False).input_ids

        if len(self.yes_ids) == 0 or len(self.no_ids) == 0:
            raise ValueError("Tokenizer produced empty ids for yes/no labels.")

    def build_prompt(self, question: str, answer: str) -> str:
        """
        Build the verifier prompt.
        IMPORTANT: This should be used consistently in both inference and training.
        """
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Is this answer correct? Answer y(Yes) or n(No)."
        )
        return prompt

    @torch.no_grad()
    def score(self, question: str, answer: str) -> float:
        """
        Return P(correct | question, answer) in (0, 1).

        Optimized implementation:
        1. Uses Batching (computes Yes and No in a single forward pass).
        2. Handles Tokenization correctly by concatenating IDs instead of strings.
        """
        # 1. Prepare Prompt IDs
        prompt = self.build_prompt(question, answer)
        # Add BOS token if the model expects it, but do not truncate here generally
        context_enc = self.tokenizer(prompt, add_special_tokens=True)
        context_ids = context_enc.input_ids

        # 2. Prepare Sequences (Context + Label) via Tensor Concatenation
        # We convert to tensor immediately to use efficient concatenation
        device = self.model.device
        ctx_tensor = torch.tensor(context_ids, dtype=torch.long, device=device)
        yes_tensor = torch.tensor(self.yes_ids, dtype=torch.long, device=device)
        no_tensor = torch.tensor(self.no_ids, dtype=torch.long, device=device)

        # Create two sequences: [Context, label_yes] and [Context, label_no]
        seq_yes = torch.cat([ctx_tensor, yes_tensor])
        seq_no = torch.cat([ctx_tensor, no_tensor])

        # 3. Batching and Padding
        # Pad sequences to handle cases where 'label_yes' and 'label_no' differ in length
        pad_val = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence([seq_yes, seq_no], batch_first=True, padding_value=pad_val)
        attention_mask = (input_ids != pad_val).long()

        # 4. Forward Pass (Batch Size = 2)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: [2, seq_len, vocab_size]
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)

        # 5. Extract Log Probabilities for Labels
        # Helper to sum log-probs of the label tokens given context
        def get_label_logprob(batch_idx, label_len):
            # The label starts immediately after the context
            start_pos = len(context_ids)
            end_pos = start_pos + label_len
            total_logprob = 0.0
            for _, pos in enumerate(range(start_pos, end_pos)):
                target_token_id = input_ids[batch_idx, pos].item()
                # To predict token at `pos`, we look at logits at `pos - 1`
                token_logprob = log_probs[batch_idx, pos - 1, target_token_id].item()
                total_logprob += token_logprob
            return total_logprob

        logp_yes = get_label_logprob(0, len(self.yes_ids))
        logp_no = get_label_logprob(1, len(self.no_ids))

        # 6. Normalize: P(Yes) = exp(Yes) / (exp(Yes) + exp(No))
        max_logp = max(logp_yes, logp_no)
        p_yes_score = math.exp(logp_yes - max_logp)
        p_no_score = math.exp(logp_no - max_logp)
        prob_correct = p_yes_score / (p_yes_score + p_no_score)
        return float(prob_correct)
