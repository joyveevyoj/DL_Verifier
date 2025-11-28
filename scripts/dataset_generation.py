import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
# Test with the first four questions from the test set
import time
import re
import json

def extract_answer(text):
    """
    Extract the numerical answer from the text.
    GSM8K answers typically end with #### followed by the number.
    """
    # Try to find the answer after ####
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        # Remove commas from the number
        return match.group(1).replace(',', '')
    
    # Fallback: try to find the last number in the text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return None

def check_answer_correct(generated_answer, reference_answer):
    """
    Check if the generated answer matches the reference answer.
    """
    gen = extract_answer(generated_answer)
    ref = extract_answer(reference_answer)
    
    if gen is None or ref is None:
        return False
    
    try:
        # Compare as floats to handle different formats
        return abs(float(gen) - float(ref)) < 0.01
    except:
        return gen == ref

def generate_answers(
    questions,
    tokenizer,
    model,
    num_answers=10,
    max_new_tokens=512,
    temperature=0.7,
    correct_answer=None
):
    """
    Generate multiple answers per question using Qwen chat template.
    Supports batching: provide a single question (str) or a list of questions.
    Returns a flat list if a single question is provided, otherwise a list of lists
    (answers per question).
    """
    # Normalize inputs to lists
    if isinstance(questions, str):
        question_list = [questions]
        single_question = True
    else:
        question_list = list(questions)
        single_question = False
    
    if correct_answer is None:
        correct_list = [None] * len(question_list)
    elif isinstance(correct_answer, str):
        correct_list = [correct_answer]
    else:
        correct_list = list(correct_answer)
    
    if len(correct_list) != len(question_list):
        raise ValueError("Length of correct_answer list must match number of questions")
    
    # Build prompts per question
    prompt_texts = []
    for q in question_list:
        prompt = f"""Question: {q}
Answer: Let's solve this step by step concisely. End your answer with #### followed by the final numerical answer."""
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompt_texts.append(text)
    
    # Tokenize batch
    model_inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)
    attention_mask = model_inputs.get("attention_mask", None)
    input_lengths = attention_mask.sum(dim=1).tolist() if attention_mask is not None else [len(ids) for ids in model_inputs["input_ids"]]
    
    batch_size = len(question_list)
    print(f"\nGenerating {num_answers} answers for each of {batch_size} questions (total {batch_size * num_answers})...")
    print('='*60)
    
    # Generate sequences in parallel
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            num_return_sequences=num_answers,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode outputs grouped by question
    grouped_answers = []
    for q_idx, question in enumerate(question_list):
        question_answers = []
        print(f"\n{'='*60}")
        print(f"Question {q_idx + 1}: {question}")
        print(f"Reference: {extract_answer(correct_list[q_idx]) if correct_list[q_idx] else 'N/A'}")
        print('-'*60)
        for ans_idx in range(num_answers):
            seq_idx = q_idx * num_answers + ans_idx
            prompt_len = input_lengths[q_idx]
            output_ids = generated_ids[seq_idx][prompt_len:].tolist()
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            question_answers.append(generated_text)
            print(f"Generated Answer {ans_idx + 1}:")
            print('-'*60)
            print(generated_text)
            print('-'*60)
            extracted = extract_answer(generated_text)
            print(f"Extracted Answer: {extracted}")
            if correct_list[q_idx] is not None:
                correct_extracted = extract_answer(correct_list[q_idx])
                print(f"Correct Answer: {correct_extracted}")
            print('-'*60)
        grouped_answers.append(question_answers)
    
    return grouped_answers[0] if single_question else grouped_answers


def save_verifier_dataset(output_path, questions, generated_answers, reference_answers):
    """
    Save verifier dataset where each record bundles all answers for a question.
    Record format:
        {
            "question": str,
            "reference_answer": str,
            "answers": [str, ...],
            "answer_labels": [0|1, ...],  # 1 == correct
            "total_answers_num": int,
            "correct_answers_num": int
        }
    """
    records = []
    for question, answers, reference in zip(questions, generated_answers, reference_answers):
        labels = [1 if check_answer_correct(ans, reference) else 0 for ans in answers]
        record = {
            "question": question,
            "reference_answer": reference,
            "answers": answers,
            "answer_labels": labels,
            "total_answers_num": len(answers),
            "correct_answers_num": sum(labels)
        }
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved verifier dataset with {len(records)} records to {output_path}")


def main():
    # Load the tokenizer and the model
    model_name = "Qwen/Qwen3-0.6B"

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main")

    print(f"Dataset loaded: {len(ds['test'])} test examples, {len(ds['train'])} train examples")
    print(f"Model loaded on device: {model.device}")
    num_questions = 1
    num_answers = 32

    test_examples = [ds['test'][i] for i in range(num_questions)]
    # Keep the first example for downstream cells that expect `test_example`
    test_example = test_examples[0]
    questions = [example['question'] for example in test_examples]
    reference_answers = [example['answer'] for example in test_examples]
    reference_final = [extract_answer(ans) for ans in reference_answers]

    for idx, example in enumerate(test_examples):
        print(f"Question {idx + 1}: {example['question']}")
        print(f"\nReference Answer: {example['answer']}")
        print(f"Reference Final Answer: {reference_final[idx]}")
        print("\n" + "-"*80)

    print("="*80)
    print(f"Generating {num_answers} answers for each of {num_questions} questions...")
    print("="*80)

    # Time the generation process
    start_time = time.time()
    generated_answers = generate_answers(
        questions,
        tokenizer,
        model,
        num_answers=num_answers,
        correct_answer=reference_answers
    )
    end_time = time.time()

    # Calculate and print timing statistics
    total_time = end_time - start_time
    total_generations = num_questions * num_answers
    avg_time_per_answer = total_time / total_generations

    print("\n" + "="*80)
    print("TIMING STATISTICS:")
    print("="*80)
    print(f"Total time for {total_generations} answers: {total_time:.2f} seconds")
    print(f"Average time per answer: {avg_time_per_answer:.2f} seconds")

    print("\n" + "="*80)
    print("VERIFICATION RESULTS:")
    print("="*80)

    total_correct = 0
    for q_idx, (example, answers) in enumerate(zip(test_examples, generated_answers)):
        print(f"\nQuestion {q_idx + 1} Results:")
        question_correct = 0
        for a_idx, answer in enumerate(answers):
            extracted = extract_answer(answer)
            is_correct = check_answer_correct(answer, example['answer'])
            # print(f"\nAnswer {a_idx + 1}:")
            # print(f"Extracted value: {extracted}")
            # print(f"Correct: {is_correct}")
            # print(f"Response preview: {answer}...")
            if is_correct:
                question_correct += 1
                total_correct += 1
        print(f"\nSummary for Question {q_idx + 1}: {question_correct}/{num_answers} answers were correct")

    print("\n" + "="*80)
    print(f"Overall: {total_correct}/{total_generations} answers were correct across all questions")

    # Persist verifier dataset
    output_path = "verifier_dataset.json"
    save_verifier_dataset(output_path, questions, generated_answers, reference_answers)



if __name__ == "__main__":
    main()

    