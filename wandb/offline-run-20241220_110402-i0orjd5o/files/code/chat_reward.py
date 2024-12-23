import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from datasets import load_dataset, Dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    get_scheduler,
)
from huggingface_hub import HfApi

api = HfApi()


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = False
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 220
    """How often to print sample output"""
    run_eval: bool = False
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    gradient_accumulation_steps: int = 8
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = 92832
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_eval_batch_size: int = 1
    """per rank eval batch size"""

    # other args
    base_model: str = "EleutherAI/pythia-160m"
    """the name of the pretrained model to use"""
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    """the query dataset"""
    reward_model_path: str = ""
    """the path to the reward model"""
    sft_model_path: str = "EleutherAI/pythia-160m"
    """the path to the sft model"""
    label_dataset: str = "vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""

    # wandb and HF tracking configs
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    push_to_hub: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """the user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """the id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """the revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """the url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "models/reward_model"
    """Where to save the model"""


def parse_args() -> Args:
    args = tyro.cli(Args)
    # Simplify logic since no accelerate
    args.world_size = 1
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps
    args.micro_batch_size = args.local_micro_batch_size * args.world_size
    args.batch_size = args.local_batch_size * args.world_size
    time_int = int(time.time())
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    if args.push_to_hub:
        if args.hf_repo_id is None: # auto-generate one
            args.hf_repo_id = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        if args.hf_entity is None:  # find the current user
            args.hf_entity = api.whoami()["name"]
        if "/" not in args.hf_repo_id: # prepend the current user
            args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    return args


def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def get_reward(model, query_responses, tokenizer, context_length=0):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    return reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1)

def evaluate(args: Args, tokenizer, model, dataloader, device):
    model.eval()
    items = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(dataloader):
            data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            predicted_reward = get_reward(model, query_responses, tokenizer)
            chosen_rewards = predicted_reward[:data['query_chosen_token'].shape[0]]
            rejected_rewards = predicted_reward[data['query_chosen_token'].shape[0]:]
            accuracy = (chosen_rewards > rejected_rewards).float()

            # Since no multiple processes, just decode directly
            for i in range(len(accuracy)):
                items["query"].append(tokenizer.decode(data["query_token"][i].cpu(), skip_special_tokens=True))
                items["chosen"].append(tokenizer.decode(data["chosen_token"][i].cpu()))
                items["rejected"].append(tokenizer.decode(data["rejected_token"][i].cpu()))
                items["batch"].append(data["batch"][i].cpu().item())
                items["split"].append(data["split"][i].cpu().item())
                items["confidence"].append(data["extra.confidence"][i].cpu().item())
                items["choice"].append(data["choice"][i].cpu().item())
                items["policies"].append(data["policies"][i])
                items["chosen_policy"].append(data["chosen_policy"][i])
                items["rejected_policy"].append(data["rejected_policy"][i])
                items["accuracy"].append(accuracy[i].cpu().item())
                items["chosen_rewards"].append(chosen_rewards[i].cpu().item())
                items["rejected_rewards"].append(rejected_rewards[i].cpu().item())
    model.train()
    return pd.DataFrame(items)


if __name__ == "__main__":
    args = parse_args()

    # Set device preference: MPS -> CUDA -> CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    local_seed = args.seed

    # load dataset
    dataset = load_dataset(args.label_dataset, split="train")
    dataset = dataset.shuffle(seed=local_seed)
    dataset = dataset.select(range(args.total_episodes))
    dataset = dataset.with_format(
        "torch",
        columns=[
            "query_token",
            "chosen_token",
            "query_chosen_token",
            "rejected_token",
            "query_rejected_token",
            "batch",
            "split",
        ],
    )
    dataloader = DataLoader(dataset, batch_size=args.local_micro_batch_size)

    eval_datasets = {}
    for split in ["validation", "validation_cnndm"]:
        validation_dataset = load_dataset(args.label_dataset, split=split).flatten()
        validation_dataset = validation_dataset.with_format(
            "torch",
            columns=[
                "query_token",
                "choice",
                "chosen_token",
                "query_chosen_token",
                "rejected_token",
                "query_rejected_token",
                "batch",
                "split",
                "extra.confidence",
                "chosen_policy",
                "rejected_policy",
                "policies",
            ],
        )
        eval_datasets[split] = validation_dataset
        print(f"The number of samples in {split}", len(validation_dataset))
    print("The number of samples in dataset", len(dataset))
    args.total_episodes = len(dataset)
    args.num_updates = args.total_episodes // args.batch_size

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    console = Console(force_terminal=True)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=asdict(args),
            name=args.run_name,
            save_code=True,
        )
        file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
        wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    pprint(args)

    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    model_config = AutoConfig.from_pretrained(args.sft_model_path)
    scalar_model_config = ScalarModelConfig(
        base_model=args.sft_model_path,
        base_config=model_config,
        hidden_size=model_config.hidden_size,
    )
    if len(args.reward_model_path) == 0:
        model: PreTrainedModel = ScalarModel(scalar_model_config)
    else:
        model: PreTrainedModel = ScalarModel.from_pretrained(
            args.reward_model_path,
            trust_remote_code=True,
        )
    disable_dropout(model)
    pprint(model_config)

    model.to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)

    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_updates * args.num_train_epochs,
    )

    print("===training model===")
    model.train()
    gradient_accumulation_idx = 0
    global_step = 0
    update = 0
    losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
    accuracies = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_preferreds = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_rejecteds = torch.zeros((args.gradient_accumulation_steps,), device=device)

    for epoch in range(args.num_train_epochs):
        print(f"epoch: {epoch}")
        for data in dataloader:
            data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
            update += 1
            global_step += args.micro_batch_size
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            predicted_reward = get_reward(model, query_responses, tokenizer)
            chosen_rewards = predicted_reward[:data['query_chosen_token'].shape[0]]
            rejected_rewards = predicted_reward[data['query_chosen_token'].shape[0]:]
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            loss.backward()

            
            gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
            if update > 1 and (update - 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # Log
                losses[gradient_accumulation_idx] = loss
                accuracies[gradient_accumulation_idx] = accuracy
                reward_preferreds[gradient_accumulation_idx] = chosen_rewards.mean()
                reward_rejecteds[gradient_accumulation_idx] = rejected_rewards.mean()
                
                train_accuracy = accuracies.mean().item()
                writer.add_scalar("train/rm/loss", losses.mean().item(), global_step)
                writer.add_scalar("train/rm/accuracy", train_accuracy, global_step)
                writer.add_scalar(
                    "train/rm/chosen_rewards", reward_preferreds.mean().item(), global_step
                )
                writer.add_scalar("train/rm/rejected_rewards", reward_rejecteds.mean().item(), global_step)
                writer.add_scalar("train/rm/lr", scheduler.get_last_lr()[0], global_step)
                pprint(
                    f"{train_accuracy=}, {scheduler.get_last_lr()=}, {optimizer.param_groups[0]['lr']=}, {update=}"
                )
    if args.run_eval:
        for eval_split, eval_dataset in eval_datasets.items():
            eval_dataloader = DataLoader(eval_dataset, batch_size=args.local_eval_batch_size)
            eval_df = evaluate(args, tokenizer, model, eval_dataloader, device)
            for split_val, row in eval_df[["split", "accuracy"]].groupby(["split"]).mean().iterrows():
                writer.add_scalar(f"eval/rm/{eval_split}/accuracy/split/{split_val}", row["accuracy"], global_step)
                print(f"eval/rm/{eval_split}/accuracy/split/{split_val}: {row['accuracy']}")
            for batch, row in eval_df[["batch", "accuracy"]].groupby(["batch"]).mean().iterrows():
                writer.add_scalar(f"eval/rm/{eval_split}/accuracy/batch/{batch}", row["accuracy"], global_step)
                print(f"eval/rm/{eval_split}/accuracy/batch/{batch}: {row['accuracy']}")
            for confi, row in eval_df[["confidence", "accuracy"]].groupby(["confidence"]).mean().iterrows():
                writer.add_scalar(f"eval/rm/{eval_split}/accuracy/confidence/{confi}", row["accuracy"], global_step)
                print(f"eval/rm/{eval_split}/accuracy/confidence/{confi}: {row['accuracy']}")
            writer.add_scalar(f"eval/rm/{eval_split}/accuracy", eval_df["accuracy"].mean(), global_step)
            print(f"eval/rm/{eval_split}/accuracy: {eval_df['accuracy'].mean()}")
            os.makedirs(f"eval_tables/{args.run_name}", exist_ok=True)
            eval_df.to_csv(f"eval_tables/{args.run_name}/eval_{eval_split}_{update}.csv")
            if args.track:
                import wandb
                wandb.log({f"samples/{eval_split}/query_responses": wandb.Table(dataframe=eval_df)}, step=update)
            del eval_df
            torch.cuda.empty_cache()

    norm_dataset = load_dataset(args.query_dataset, split="train")
    norm_dataset = norm_dataset.with_format("torch", columns=["query_token", "reference_response_token", "query_reference_response_token"])
    norm_dataset = norm_dataset.shuffle(seed=local_seed)
    norm_dataloader = DataLoader(norm_dataset, batch_size=args.local_eval_batch_size)
    items = defaultdict(list)
    rtol = 1e-2
    with torch.no_grad():
        for data in tqdm(norm_dataloader):
            data = {k: v.to(device) if torch.is_tensor(v) else v for k,v in data.items()}
            reference_responses = data["reference_response_token"]
            queries = data["query_token"]
            query_responses = data["query_reference_response_token"]
            cat_query_responses = torch.cat((queries, reference_responses), dim=1)
            cat_predicted_reward = get_reward(model, cat_query_responses, tokenizer, context_length=queries.shape[1])
            predicted_reward = get_reward(model, query_responses, tokenizer)
            unexpecte_reward_diff = predicted_reward - cat_predicted_reward
            unexpecte_reward_diff_gt_rtol = unexpecte_reward_diff.abs() > rtol

            for i in range(len(predicted_reward)):
                items["query"].append(tokenizer.decode(queries[i].cpu(), skip_special_tokens=True))
                items["reference_response"].append(tokenizer.decode(reference_responses[i].cpu()))
                items["predicted_reward"].append(predicted_reward[i].cpu().item())
                items["unexpecte_reward_diff"].append(unexpecte_reward_diff[i].cpu().item())
                items["unexpecte_reward_diff_gt_rtol"].append(unexpecte_reward_diff_gt_rtol[i].cpu().item())

    norm_df = pd.DataFrame(items)
    os.makedirs(f"eval_tables/{args.run_name}", exist_ok=True)
    norm_ds = Dataset.from_pandas(norm_df)
    norm_ds.save_to_disk(f"eval_tables/{args.run_name}/eval_{update}_normalized")
    if args.track:
        wandb.log({"samples/normalized": wandb.Table(dataframe=norm_df)}, step=update)
    stats = {
        "mean": norm_df["predicted_reward"].mean(),
        "std": norm_df["predicted_reward"].std(),
        "max": norm_df["predicted_reward"].max(),
        "min": norm_df["predicted_reward"].min(),
        "unexpecte_reward_diff_mean": norm_df["unexpecte_reward_diff"].mean(),
        "unexpecte_reward_diff_gt_rtol_mean": norm_df["unexpecte_reward_diff_gt_rtol"].mean(),
    }
    for stat_name, stat_value in stats.items():
        writer.add_scalar(f"eval/rm/normalized_{stat_name}", stat_value, global_step)
        print(f"Normalized Reward {stat_name.capitalize()}: {stat_value}")

    # save model
    if args.output_dir and args.num_train_epochs > 0:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        tokenizer.save_pretrained(args.output_dir)
        model.config.bias = norm_df["predicted_reward"].mean()
        model.save_pretrained(
            args.output_dir,
            safe_serialization=False,
        )
        if args.push_to_hub:
            model.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision, safe_serialization=False)
            print(f"🔥 pushed to {args.hf_repo_url}")
