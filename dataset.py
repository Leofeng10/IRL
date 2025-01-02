import copy
import multiprocessing
import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import tyro
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi
from rich.pretty import pprint
from transformers import AutoTokenizer


api = HfApi()

@dataclass
class TaskQueryHParams:
    """
    Hyperparameters controlling how we format, pad, and truncate
    the 'query' prompt. Also dictates max token lengths for 
    supervised fine-tuning (SFT) and reward model (RM) usage.
    """
    length: Optional[int] = None
    format_str: Optional[str] = None
    truncate_field: Optional[str] = None
    truncate_text: Optional[str] = None
    padding: Optional[Literal["empty_space", "pad_token"]] = None
    pad_token: Optional[str] = None
    pad_side: Optional[str] = None
    max_sft_response_length: Optional[int] = None
    max_sft_query_response_length: Optional[int] = None
    max_rm_response_length: Optional[int] = None
    max_rm_query_response_length: Optional[int] = None

@dataclass
class Args:
    """
    Main command-line arguments for controlling:
      - Base model
      - Hugging Face entity/user to push to
      - Whether or not to push the dataset to the Hub
      - Debug toggles
      - TL;DR-like parameters (for query truncation/padding)
    """
    base_model: str = "EleutherAI/pythia-1b-deduped"
    hf_entity: Optional[str] = None
    push_to_hub: bool = False
    check_length_correctness: bool = True
    debug: bool = False
    dataset: str = "gsm8k"

    tldr_params: TaskQueryHParams = field(
        default_factory=lambda: TaskQueryHParams(
            length=512,
            # For GSM8K, let's label it as "QUESTION:" -> "SOLUTION:"
            format_str="QUESTION: {question}\n\nSOLUTION:",
            truncate_field="question",
            truncate_text="\n",
            padding="pad_token",
            pad_side="left",
            max_sft_response_length=100,         
            max_sft_query_response_length=1000,  
            max_rm_response_length=250,         # If you need RM usage
            max_rm_query_response_length=1000,   # If you need RM usage
        )
    )

def _ensure_length(tokens, desired_length, pad_sequence=None, pad_side=None, truncate_side=None):
    """
    Ensures 'tokens' is exactly 'desired_length' in size.
    If tokens is shorter, pad either left or right with 'pad_sequence';
    if tokens is longer, truncate from the left or right.
    """
    assert pad_side in (None, "left", "right"), "Invalid pad_side."
    assert truncate_side in (None, "left", "right"), "Invalid truncate_side."

    # Pad if needed
    if len(tokens) < desired_length:
        if pad_side is None:
            # We do not expect to pad at all
            assert len(tokens) == desired_length, f"Needed to pad! {len(tokens)} < {desired_length}"
            return tokens
        pad_amt = desired_length - len(tokens)
        assert pad_sequence is not None, "No pad sequence provided."
        assert len(pad_sequence) >= pad_amt, f"Pad sequence too short ({len(pad_sequence)} < {pad_amt})"
        if pad_side == "left":
            return pad_sequence[-pad_amt:] + tokens
        else:
            # pad_side == "right"
            return tokens + pad_sequence[:pad_amt]

    # Truncate if needed
    if len(tokens) > desired_length:
        if truncate_side is None:
            raise ValueError(f"Needed to truncate! {len(tokens)} > {desired_length} but no truncate_side provided.")
        if truncate_side == "left":
            return tokens[-desired_length:]
        else:
            return tokens[:desired_length]

    # If it's exactly the desired length, do nothing
    return tokens

def _get_query_padding_for_task(encoder, hparams: TaskQueryHParams):
    """
    Returns a padding sequence for 'hparams.length' tokens.
    If hparams.padding == "empty_space", pad with tokens
    corresponding to ' ' (space). Otherwise use the
    model's pad token.
    """
    return hparams.pad_token * hparams.length

def process_query(query_info: Dict[str, str], *, encoder, hparams: TaskQueryHParams, pad_sequence=None):
    """
    Given a dict like {"question": "..."},
    produce a truncated/padded 'query' string based on hparams.format_str.
    """
    if pad_sequence is None:
        pad_sequence = _get_query_padding_for_task(encoder, hparams)

    # Convert any plain string input to a dict:
    if isinstance(query_info, str):
        query_info = dict(question=query_info)
    else:
        query_info = dict(query_info)  # shallow copy

    # Format the text according to hparams
    format_str = hparams.format_str or "{question}"
    query_text = format_str.format(**query_info)
    query_tokens = encoder.encode(query_text)

    # We'll truncate from 'truncate_field'
    truncate_field = hparams.truncate_field or "question"
    if truncate_field not in query_info:
        raise ValueError(f"Cannot truncate '{truncate_field}' in {query_info.keys()}")

    # Keep reducing the text until it fits within hparams.length tokens
    while len(query_tokens) > hparams.length:
        if not len(query_info[truncate_field]):
            raise ValueError("Ran out of text to truncate!")
        # Remove from the end or try to break at hparams.truncate_text
        i = -1  # default: chop off last character
        if hparams.truncate_text:
            try:
                i = query_info[truncate_field].rindex(hparams.truncate_text)
            except ValueError:
                pass
        query_info[truncate_field] = query_info[truncate_field][:i]
        query_text = format_str.format(**query_info)
        query_tokens = encoder.encode(query_text)

    # Now pad or confirm we have exactly hparams.length tokens
    query_token = _ensure_length(
        query_tokens,
        hparams.length,
        pad_side=hparams.pad_side,
        pad_sequence=pad_sequence
    )

    # Get the final truncated/padded string
    query_str = encoder.decode(query_token, skip_special_tokens=True).lstrip()
    return {"query_token": query_token, "query": query_str}

def find_max_token_length(dataset,encoder):
    """
    Find the maximum token length in a dataset of question-answer pairs.
    
    Args:
        dataset: HuggingFace dataset with 'question' and 'answer' columns
        model_name: Name of the pretrained model/tokenizer to use
    
    Returns:
        dict: Contains max length, example with max length, and split it came from
    """
    
    max_length = 0
    max_example = {}
    
    # Process both train and test splits
    for split in ['train', 'test']:
        split_data = dataset[split]
        
        # Process each example
        for example in split_data:
            # Combine question and answer
            combined_text = example['question'] + " " + example['answer']
            
            # Tokenize
            tokens = encoder.encode(combined_text)
            
            # Update max if this is longer
            if len(tokens) > max_length:
                max_length = len(tokens)
                max_example = {
                    'text': combined_text,
                    'length': len(tokens),
                    'split': split,
                    'question': example['question'],
                    'answer': example['answer']
                }
    
    return {
        'max_length': max_length,
        'max_example': max_example
    }


def process_batch(examples, encoder, hparams):
    """
    Batch version of the data processing logic, returning all new columns.
    """
    out_reference_response = []
    out_reference_response_token = []
    out_reference_response_token_len = []
    out_query = []
    out_query_token = []
    out_query_reference_response = []
    out_query_reference_response_token = []
    out_query_reference_response_token_len = []
    out_query_reference_response_token_response_label = []

    # Prepare a pad sequence for queries (to avoid repeatedly computing it).
    pad_sequence = _get_query_padding_for_task(encoder, hparams)

    for question, answer in zip(examples["question"], examples["answer"]):
        # Create reference_response
        reference_response = f" {answer}<|endoftext|>"

        # Build the truncated/padded query
        processed = process_query(
            {"question": question},
            encoder=encoder,
            hparams=hparams,
            pad_sequence=pad_sequence
        )
        # reference_response_token
        reference_response_token = encoder.encode(
            reference_response,
            padding="max_length",
            max_length=hparams.max_sft_response_length,
            truncation=True,
        )
        reference_response_token_len = len(
            encoder.encode(reference_response, add_special_tokens=False)
        )

        # Build query + reference response string
        query_reference_response = processed["query"].strip() + reference_response
        # If we want to ensure the maximum length for the combined string:
        if hparams.padding == "empty_space":
            query_reference_response_token = (
                processed["query_token"] + reference_response_token
            )
        else:
            query_reference_response_token = encoder.encode(
                query_reference_response,
                padding="max_length",
                max_length=hparams.max_sft_query_response_length,
                truncation=True,
            )
        query_reference_response_token_len = len(
            encoder.encode(query_reference_response, add_special_tokens=False)
        )

        # Create label array that masks out the prompt portion
        unpadded_query_token = [
            tok for tok in processed["query_token"]
            if tok != encoder.pad_token_id
        ]
        label_tokens = copy.deepcopy(query_reference_response_token)
        for i in range(len(unpadded_query_token)):
            label_tokens[i] = encoder.pad_token_id

        out_reference_response.append(reference_response)
        out_reference_response_token.append(reference_response_token)
        out_reference_response_token_len.append(reference_response_token_len)
        out_query.append(processed["query"])
        out_query_token.append(processed["query_token"])
        out_query_reference_response.append(query_reference_response)
        out_query_reference_response_token.append(query_reference_response_token)
        out_query_reference_response_token_len.append(query_reference_response_token_len)
        out_query_reference_response_token_response_label.append(label_tokens)

    return {
        "reference_response": out_reference_response,
        "reference_response_token": out_reference_response_token,
        "reference_response_token_len": out_reference_response_token_len,
        "query": out_query,
        "query_token": out_query_token,
        "query_reference_response": out_query_reference_response,
        "query_reference_response_token": out_query_reference_response_token,
        "query_reference_response_token_len": out_query_reference_response_token_len,
        "query_reference_response_token_response_label": out_query_reference_response_token_response_label,
    }

if __name__ == "__main__":
    ############################################################
    # 0) Parse arguments
    ############################################################
    args = tyro.cli(Args)

    # Determine user or org name if not provided
    if args.hf_entity is None:
        args.hf_entity = api.whoami()["name"]
        assert isinstance(args.hf_entity, str), "hf_entity must be a string."

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Configure padding tokens
    if args.tldr_params.padding == "empty_space":
        args.tldr_params.pad_token = tokenizer.encode(" ")
    else:
        args.tldr_params.pad_token = [tokenizer.pad_token_id]

    pprint(args)
    timestamp = int(time.time())

    ############################################################
    # 1) Load GSM8K dataset & add a validation split
    ############################################################
    gsm8k = load_dataset(args.dataset, "main")  ## IF Error: Use ignore_verifications=True

    # find the max tokenized length between all splits
    max_token_len = find_max_token_length(gsm8k,tokenizer)
    args.tldr_params.max_sft_query_response_length = max_token_len['max_length'] + 50  #additional 50 tokens for sanity
    # Create a train/validation split from the original train
    train_val = gsm8k["train"].train_test_split(test_size=0.1, seed=42)
    # Overwrite gsm8k to include train, validation, test
    gsm8k_ds = {
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": gsm8k["test"],
    }

    ############################################################
    # 2) Process the dataset splits
    ############################################################
    num_proc = 1 if args.debug else multiprocessing.cpu_count()

    sft_ds = {}
    for split_name, split_data in gsm8k_ds.items():
        sft_ds[split_name] = split_data.map(
            lambda batch: process_batch(batch, tokenizer, args.tldr_params),
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=False,
            desc=f"Processing {split_name} split",
        )

    # Convert dict to a DatasetDict
    final_ds = DatasetDict(sft_ds)

    ############################################################
    # 3) Optionally push to the Hugging Face Hub
    ############################################################
    if args.push_to_hub:
        final_ds.push_to_hub(
            repo_id=f"{args.hf_entity}/gsm8k_tldr_style",
            private=True  # Set to False if you want it public
        )
        print("Dataset pushed to the Hub!")

    print("All done!")
