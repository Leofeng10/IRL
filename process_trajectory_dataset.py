# process_trajectory_dataset.py

import json
import os
from typing import List, Dict
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data


def build_dialogue_text(goal: str, trajectory: List[Dict]) -> str:
    dialogue = f"[GOAL] {goal}\n"
    for turn in trajectory:
        speaker = turn['speaker'].upper()
        text = turn['text'].strip()
        dialogue += f"[{speaker}] {text}\n"
    return dialogue.strip()


def preprocess_dataset(
    path_to_jsonl: str,
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 1024
) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    raw_data = load_jsonl(path_to_jsonl)
    processed = {
        "dialogue_text": [],
        "input_ids": [],
        "attention_mask": [],
        "label": []  # 1 if source == "sft", 0 if source == "init"
    }

    print("Tokenizing...")
    for item in tqdm(raw_data):
        text = build_dialogue_text(item['goal'], item['trajectory'])
        label = 1 if item.get("source", "init") == "sft" else 0
        tokenized = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        processed["dialogue_text"].append(text)
        processed["input_ids"].append(tokenized['input_ids'][0])
        processed["attention_mask"].append(tokenized['attention_mask'][0])
        processed["label"].append(label)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(processed)
    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to your .jsonl file")
    parser.add_argument("--output_path", type=str, default="processed_dataset", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="Tokenizer name")
    parser.add_argument("--max_length", type=int, default=1024, help="Max token length")
    args = parser.parse_args()

    dataset = preprocess_dataset(args.input_path, args.tokenizer, args.max_length)
    os.makedirs(args.output_path, exist_ok=True)
    dataset.save_to_disk(args.output_path)
    print(f"✅ Dataset saved to {args.output_path}")
