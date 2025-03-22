# ppo_finetune_with_irl.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from tqdm import tqdm

from train_irl_trajectory import TrajectoryRewardModel


def compute_log_probs(model, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs


def ppo_update(policy_model, old_log_probs, input_ids, attention_mask, rewards, optimizer, clip_epsilon=0.2):
    policy_model.train()
    outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits
    new_log_probs = torch.log_softmax(logits, dim=-1)

    # Shift for decoder-style models
    shifted_input_ids = input_ids[:, 1:].contiguous()
    new_log_probs = new_log_probs[:, :-1, :].contiguous()
    old_log_probs = old_log_probs[:, :-1, :].contiguous()
    attention_mask = attention_mask[:, 1:].contiguous()

    # Gather log_probs of taken actions (next tokens)
    new_log_probs_act = new_log_probs.gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    old_log_probs_act = old_log_probs.gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

    # Advantage: reward - baseline (no baseline here, so raw reward)
    advantage = rewards.unsqueeze(-1).expand_as(new_log_probs_act)

    # Ratio for PPO
    ratio = torch.exp(new_log_probs_act - old_log_probs_act)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

    # Only compute loss over non-padding tokens
    mask = attention_mask.float()
    loss = (policy_loss * mask).sum() / mask.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def finetune_with_ppo(
    dataset_path,
    policy_model_name="gpt2",
    reward_model_path="irl_reward_model.pth",
    reward_encoder_name="bert-base-uncased",
    epochs=3,
    batch_size=4,
    lr=1e-5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load models
    policy_model = AutoModelForCausalLM.from_pretrained(policy_model_name).to(device)
    reward_model = TrajectoryRewardModel(reward_encoder_name).to(device)
    reward_model.load_state_dict(torch.load(reward_model_path))
    reward_model.eval()

    optimizer = AdamW(policy_model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get rewards from reward model
            with torch.no_grad():
                reward_scores = reward_model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute old log probs
            old_log_probs = compute_log_probs(policy_model, input_ids, attention_mask)

            # PPO update
            loss = ppo_update(
                policy_model,
                old_log_probs,
                input_ids,
                attention_mask,
                reward_scores,
                optimizer
            )
            total_loss += loss

        print(f"✅ Epoch {epoch + 1} | Avg PPO Loss: {total_loss / len(dataloader):.4f}")

    policy_model.save_pretrained("ppo_finetuned_model")
    tokenizer.save_pretrained("ppo_finetuned_model")
    print("🎉 PPO fine-tuned model saved.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--policy_model_name", type=str, default="gpt2")
    parser.add_argument("--reward_model_path", type=str, default="irl_reward_model.pth")
    parser.add_argument("--reward_encoder_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    finetune_with_ppo(
        dataset_path=args.dataset_path,
        policy_model_name=args.policy_model_name,
        reward_model_path=args.reward_model_path,
        reward_encoder_name=args.reward_encoder_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
