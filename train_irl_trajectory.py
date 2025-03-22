# train_irl_trajectory.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


class TrajectoryRewardModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  # CLS token
        reward = self.classifier(cls)
        return reward.squeeze(-1)


def irl_kl_loss(reward_expert, reward_policy):
    return -torch.mean(reward_expert) + torch.logsumexp(reward_policy, dim=0)


def train_reward_model(
    dataset_path,
    model_name="bert-base-uncased",
    output_path="irl_reward_model.pth",
    batch_size=8,
    epochs=3,
    lr=1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("📂 Loading dataset from:", dataset_path)
    dataset = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TrajectoryRewardModel(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)  # 1 for expert, 0 for init

            rewards = model(input_ids=input_ids, attention_mask=attention_mask)

            reward_expert = rewards[labels == 1.0]
            reward_policy = rewards[labels == 0.0]

            if len(reward_expert) == 0 or len(reward_policy) == 0:
                continue  # skip incomplete batch

            loss = irl_kl_loss(reward_expert, reward_policy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"✅ Epoch {epoch + 1} | Avg Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"🎉 Reward model saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the processed HuggingFace dataset")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Base encoder model")
    parser.add_argument("--output_path", type=str, default="irl_reward_model.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_reward_model(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_path=args.output_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )
