import torch
from torch.utils.data import Dataset


class VerifierDataset(Dataset):
    def __init__(self, raw_data, verifier, max_length: int = 512):
        """
        Args:
            raw_data: List of dicts. Each dict contains 'question', 'reference_answer',
                      'answers' (list), and 'answer_labels' (list).
            verifier: The verifier model wrapper.
            max_length: Token limit.
        """
        self.verifier = verifier
        self.tokenizer = verifier.tokenizer
        self.max_length = max_length
        self.samples = []

        # Flatten the dataset with Data Augmentation
        for entry in raw_data:
            question = entry["question"]

            # 1. Get original answers and labels
            answers = entry["answers"]
            labels = entry["answer_labels"]

            # 2. Get the reference answer
            ref_answer = entry["reference_answer"]

            # 3. Build the complete list for training
            # If a reference answer exists, append it to the end and add the corresponding label 1
            if ref_answer is not None:
                train_answers = answers + [ref_answer]
                train_labels = labels + [1]
            else:
                print("Warning: No reference answer provided for question:", question)
                train_answers = answers
                train_labels = labels

            # 4. Flatten the dataset
            # Split the structure of 1 Question -> N+1 Answers into N+1 independent samples
            for ans, lbl in zip(train_answers, train_labels):
                self.samples.append({
                    "question": question,
                    "answer": ans,
                    "label": lbl
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        q = ex["question"]
        a = ex["answer"]
        y = ex["label"]  # 1 or 0

        # Build prompt
        prompt = self.verifier.build_prompt(q, a)

        # Label mapping
        target_text = self.verifier.label_yes if y == 1 else self.verifier.label_no

        # 1. Tokenize Context
        context_enc = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_attention_mask=False
        )
        context_ids = context_enc.input_ids

        # 2. Tokenize Label
        target_enc = self.tokenizer(
            target_text,
            add_special_tokens=False,
            return_attention_mask=False
        )
        target_ids = target_enc.input_ids
        target_ids += [self.tokenizer.eos_token_id]

        # 3. Concatenate
        full_ids = context_ids + target_ids

        # 4. Truncate
        if len(full_ids) > self.max_length:
            full_ids = full_ids[-self.max_length:]
            new_context_len = len(full_ids) - len(target_ids)
        else:
            new_context_len = len(context_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        if new_context_len > 0:
            labels[:new_context_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
