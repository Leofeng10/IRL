# ppo_trl_with_irl.py

from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import torch
from train_irl_trajectory import TrajectoryRewardModel


def build_reward_fn(reward_model_path, reward_encoder_name):
    reward_model = TrajectoryRewardModel(reward_encoder_name)
    reward_model.load_state_dict(torch.load(reward_model_path))
    reward_model.eval()
    reward_model.cuda()

    def compute_rewards(samples):
        tokenizer = AutoTokenizer.from_pretrained(reward_encoder_name)
        tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(samples, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            scores = reward_model(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask)
        return scores

    return compute_rewards


def run_trl_ppo(
    dataset_path="processed_dataset",
    policy_model_name="gpt2",
    reward_model_path="irl_reward_model.pth",
    reward_encoder_name="bert-base-uncased",
    output_dir="ppo_trl_model",
    batch_size=4,
    epochs=3,
):
    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(policy_model_name)

    dataset = load_from_disk(dataset_path)
    queries = [sample["dialogue_text"] for sample in dataset]
    dataset = [{"query": q} for q in queries]

    config = PPOConfig(
        model_name=policy_model_name,
        learning_rate=1e-5,
        batch_size=batch_size,
        mini_batch_size=batch_size,
        log_with=None,
        total_epochs=epochs,
    )

    ppo_trainer = PPOTrainer(config, model=model, tokenizer=tokenizer, dataset=dataset)
    reward_fn = build_reward_fn(reward_model_path, reward_encoder_name)

    for epoch in range(epochs):
        for batch in ppo_trainer.dataloader:
            queries = batch["query"]
            responses = ppo_trainer.generate(queries)
            rewards = reward_fn(responses)
            stats = ppo_trainer.step(queries, responses, rewards)
            print(f"[Epoch {epoch+1}] PPO step complete | reward avg: {rewards.mean().item():.4f}")

    ppo_trainer.save_pretrained(output_dir)
    print(f"✅ Saved PPO+IRL fine-tuned model to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="processed_dataset")
    parser.add_argument("--policy_model_name", type=str, default="gpt2")
    parser.add_argument("--reward_model_path", type=str, default="irl_reward_model.pth")
    parser.add_argument("--reward_encoder_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="ppo_trl_model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    run_trl_ppo(
        dataset_path=args.dataset_path,
        policy_model_name=args.policy_model_name,
        reward_model_path=args.reward_model_path,
        reward_encoder_name=args.reward_encoder_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
