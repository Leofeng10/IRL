# Language Imitation via IRL

This repository contains an implementation attempt of the paper <span style="color: #0366d6">["Imitating Language via Scalable Inverse Reinforcement Learning"](https://arxiv.org/pdf/2409.01369)</span>.

## Overview

This is an experimental codebase that aims to implement the language imitation approach described in the paper. \
The implementation uses inverse reinforcement learning (IRL) techniques to learn language models by inferring rewards from expert demonstrations.

## Acknowledgments

The supervised fine-tuning (SFT) code is adapted from <span style="color: #0366d6">[summarize_from_feedback_details](https://github.com/vwxyzjn/summarize_from_feedback_details)</span> repository.

## Implementation Details

Since the implementation is adapted from the above tldr training, I have modified the GSM8K dataset to fit the SFT dataset from above. \
If you want to run your implementation, please run the dataset.py with the following command:

```bash
    python run.py \
        --base_model "EleutherAI/pythia-1b-deduped" \
        --hf_entity None \
        --push_to_hub False \
        --check_length_correctness True \
        --debug False
        --dataset "gsm8k"
```
## Setup and Usage

1. **Clone this repository**:
    ```bash
    git clone https://github.com/sankyde/IRL.git 
    cd IRL 
2. **Install Python dependencies (e.g., via pip)**:
    ```bash
    pip install -r requirements.txt
(Optional) Set up DeepSpeed or Accelerate for distributed GPU training.

<strong>Quick Start</strong>

1. **Run Command**:  
   ```bash
   python train.py \
       --base_model="EleutherAI/pythia-160m" \
       --lr=1e-4 \
       --num_train_epochs=3 \
       --gradient_accumulation_steps=8 \
       --local_micro_batch_size=4 \
       --lambda_td=0.1 \
       --lambda_kl=0.01 \
       --use_irl=True

# Configuration Options

## Parameter Summary
<details>
  <summary><strong>General Parameters</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `exp_name` | str | (script filename) | Name of the experiment run. Defaults to the file's basename if not set. |
| `seed` | int | 1 | Random seed for reproducibility. |
| `cuda` | bool | True | Whether to use CUDA (GPU), if available. |
| `run_name` | Optional[str] | None | Unique name for this run. Useful for logging and checkpoints. |
| `load_from_cache_file` | bool | False | If True, loads dataset from a local cache during dataset.map. |
| `deepspeed` | bool | True | Enables DeepSpeed for large-scale model training. |
| `print_sample_output_freq` | int | 220 | How often (in steps) to print sample model outputs. |
| `run_eval` | bool | True | Whether to run evaluation periodically. |
| `eval_every` | int | 500 | How many training steps between each evaluation. |
</details>

<details>
  <summary><strong>IRL Parameters</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `gamma` | float | 1.0 | Discount factor for IRL. Often set to 1.0 for text tasks. |
| `lambda_td` | float | 0.1 | Weight of the temporal difference (TD) term in the IRL objective. |
| `lambda_kl` | float | 0.01 | Strength of KL regularization, preventing the policy from drifting too far from the reference model. |
| `use_kl` | bool | True | Whether to apply KL divergence between current policy and reference model. |
| `use_mle` | bool | False | Whether to include the original MLE (cross-entropy) loss. |
| `use_irl` | bool | True | Whether to enable the IRL-based objective (IQLearn). |
| `mle_steps_before_irl` | int | 0 | Number of steps to do pure MLE before switching on the IRL objective. |
</details>

<details>
  <summary><strong>Optimizer Settings</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `eps` | float | 1e-9 | Epsilon value for the optimizer (e.g., Adam/AdamW). |
| `lr` | float | 1e-4 | Base learning rate. |
| `optimizer` | Literal["adam","adamw"] | "adamw" | Which optimizer to use. |
| `scheduler` | str | "cosine" | Learning rate scheduler type (e.g., "linear", "cosine", "constant", etc.). |
| `warm_up_steps` | int | 0 | Number of warm-up steps in the scheduler. |
</details>
<details>
  <summary><strong>Batch & Training Sizes</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `world_size` | Optional[int] | None | Number of processes (GPUs) used, typically determined by Accelerate automatically. |
| `num_train_epochs` | int | 3 | Total number of epochs to run. |
| `num_updates` | Optional[int] | None | Target total number of update steps. If both num_train_epochs and num_updates are specified, training may stop earlier once one condition is met. |
| `gradient_accumulation_steps` | int | 8 | Number of updates to accumulate gradients before performing an optimizer step. |
| `local_micro_batch_size` | Optional[int] | 4 | Per-device micro-batch size before gradient accumulation. |
| `total_episodes` | Optional[int] | None | Total episodes in the dataset (optional, for scheduling or logging). |
| `micro_batch_size` | Optional[int] | None | Micro-batch size across all devices (i.e., local_micro_batch_size * world_size). |
| `local_batch_size` | Optional[int] | None | Effective batch size per device (local_micro_batch_size * gradient_accumulation_steps). |
| `batch_size` | Optional[int] | None | Global batch size across all devices (local_batch_size * world_size). |
| `local_eval_batch_size` | int | 4 | Per-device batch size for evaluation. |
</details>
<details>
  <summary><strong>Model & Dataset Arguments</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `base_model` | str | "EleutherAI/pythia-160m" | The name/path of the pretrained model on Hugging Face Hub or locally. |
| `query_dataset` | str | "sdesai/gsm8k_tldr_style" | Dataset ID or path for training data (queries & reference responses). |
| `response_length` | int | 100 | Maximum new tokens to generate. |
| `truncate_token` | Literal["eos"] | "eos" | The special token used to truncate generation. |
| `truncate_token_id` | Optional[int] | None | Token ID for truncate_token. If None, the model uses its default eos_token_id. |
| `temperature` | float | 0.7 | Sampling temperature when generating text. |
</details>
<details>
  <summary><strong>Tracking & Outputs</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `track` | bool | False | If True, enables Weights & Biases logging. |
| `wandb_project_name` | str | "tldr_summarize" | Project name used in W&B logging. |
| `wandb_entity` | Optional[str] | None | Entity/team name for W&B. |
| `push_to_hub` | bool | False | Whether to push final model to Hugging Face Hub. |
| `hf_entity` | Optional[str] | None | User or org name on Hugging Face Hub. |
| `hf_repo_id` | Optional[str] | None | Repository ID if pushing model to the Hub. |
| `hf_repo_revision` | Optional[str] | None | Specific revision/tag in the Hub repo. |
| `hf_repo_url` | Optional[str] | None | Full URL to the Hub repository. |
| `output_dir` | str | "models/sft_model" | Directory to save final model weights, tokenizer, logs, etc. |
</details>

## Notes


This is an experimental implementation and may not fully replicate all aspects of the original paper. Use with caution and expect ongoing changes and improvements.
