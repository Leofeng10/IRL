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
| `exp_name` | str | (script filename) | The name of this experiment. |
| `seed` | int | 1 | Seed of the experiment. |
| `cuda` | bool | True | Whether to use cuda if available. |
| `run_name` | Optional[str] | None | A unique name of this run. |
| `load_from_cache_file` | bool | False | Whether to load data from the local cache file in `dataset.map`. |
| `deepspeed` | bool | True | Whether to use deepspeed to train the model. |
| `run_eval` | bool | True | Whether to run evaluation. |
| `eval_every` | int | 20 | How Often to run Eval. |
| `monitor` | bool | True | Monitor the internal Chosen Q, Value, Logits, Log_PI. |
</details>

<details>
  <summary><strong>IRL Parameters</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `gamma` | float | 1.0 | The sampling temperature. |
| `lambda_td` | float | 0.1 | TD penalty weight. |
| `lambda_kl` | float | 0.01 | KL penalty weight (Colab: 0.1). |
| `use_kl` | bool | True | Use KL Divergence between a reference model and a policy model. |
| `use_mle` | bool | False | Use Original MLE Loss. |
| `use_irl` | bool | True | Use IRL Loss. |
| `mle_steps_before_irl` | int | 0 | Using MLE to stabilize training before starting IRL. |
</details>

<details>
  <summary><strong>Optimizer Settings</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `eps` | float | 1e-9 | The epsilon value for the optimizer. |
| `lr` | float | 1e-4 | The learning rate (Colab: 3e-6). |
| `optimizer` | Literal["adam","adamw"] | "adamw" | Which optimizer to use. |
| `scheduler` | str | "cosine" | Which scheduler to use. |
| `warm_up_steps` | int | 0 | Number of warm up steps for the scheduler. |
</details>
<details>
  <summary><strong>Batch & Training Sizes</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `world_size` | Optional[int] | None | The number of processes (GPUs) to use. |
| `num_train_epochs` | int | 3 | Number of epochs to train. |
| `num_updates` | Optional[int] | None | The number of updates to train. |
| `gradient_accumulation_steps` | int | 4 | The number of gradient accumulation steps (Colab: 32). |
| `local_micro_batch_size` | Optional[int] | 2 | The micro batch size per GPU (HF's `per_device_train_batch_size`) (Colab: 4). |
| `total_episodes` | Optional[int] | None | The total number of episodes in the dataset. |
| `micro_batch_size` | Optional[int] | None | The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`). |
| `local_batch_size` | Optional[int] | None | The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`). |
| `batch_size` | Optional[int] | None | The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`). |
| `local_eval_batch_size` | int | 4 | Per rank eval batch size (Colab: 16). |
</details>
<details>
  <summary><strong>Model & Dataset Arguments</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `base_model` | str | "EleutherAI/pythia-160m" | The name of the pretrained model to use. |
| `query_dataset` | str | "sdesai/gsm8k_tldr_style" | The query dataset. |
| `response_length` | int | 500 | The length of the response. |
| `truncate_token` | Literal["eos"] | "eos" | The truncate token. |
| `truncate_token_id` | Optional[int] | None | The truncation token id. |
| `temperature` | float | 0.7 | The sampling temperature (Colab: 0.6). |
| `eval_num_samples` | int | 30 | Number of samples to choose from the eval dataset. Use Shuffle=True to make this random. |
</details>
<details>
  <summary><strong>Tracking & Outputs</strong></summary>

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `track` | bool | True | If toggled, this experiment will be tracked with Weights and Biases. |
| `wandb_project_name` | str | "IRL Finetuning" | The wandb's project name. |
| `wandb_entity` | Optional[str] | None | The entity (team) of wandb's project. |
| `push_to_hub` | bool | False | Whether to upload the saved model to huggingface. |
| `hf_entity` | Optional[str] | None | The user or org name of the model repository from the Hugging Face Hub. |
| `hf_repo_id` | Optional[str] | None | The id of the saved model in the Hugging Face Hub (can be autoset if not given). |
| `hf_repo_revision` | Optional[str] | None | The revision of the saved model in the Hugging Face Hub (can be autoset if not given). |
| `hf_repo_url` | Optional[str] | None | The url of the saved model in the Hugging Face Hub (will be autoset). |
| `output_dir` | str | "models/sft_model" | Where to save the model. |

</details>

## Notes


This is an experimental implementation and may not fully replicate all aspects of the original paper. Use with caution and expect ongoing changes and improvements.
