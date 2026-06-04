"""Compute reward-model, DPO, IPO, and GRPO-style quantities on toy logits.

Run:
    conda run -n aigc python dpo_loss_demo.py --beta 0.1
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from common import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sequence_logprob(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Return summed log p(tokens) for each sequence."""
    log_probs = logits.log_softmax(dim=-1)
    gathered = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    return gathered.sum(dim=-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pi_margin = policy_chosen - policy_rejected
    ref_margin = ref_chosen - ref_rejected
    logits = beta * (pi_margin - ref_margin)
    return -F.logsigmoid(logits).mean(), logits


def ipo_loss(logits: torch.Tensor, beta: float) -> torch.Tensor:
    target_margin = 1.0 / (2.0 * beta)
    return ((logits / beta - target_margin) ** 2).mean()


def reward_model_pair_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(chosen_scores - rejected_scores).mean()


def grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-8)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    batch_size = 4
    seq_len = 6
    vocab_size = 12
    chosen_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    ref_chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
    ref_rejected_logits = torch.randn(batch_size, seq_len, vocab_size)

    policy_chosen_logits = ref_chosen_logits.clone()
    policy_rejected_logits = ref_rejected_logits.clone()
    policy_chosen_logits.scatter_add_(
        -1,
        chosen_tokens.unsqueeze(-1),
        torch.full((batch_size, seq_len, 1), 0.35),
    )
    policy_rejected_logits.scatter_add_(
        -1,
        rejected_tokens.unsqueeze(-1),
        torch.full((batch_size, seq_len, 1), -0.15),
    )

    policy_chosen = sequence_logprob(policy_chosen_logits, chosen_tokens)
    policy_rejected = sequence_logprob(policy_rejected_logits, rejected_tokens)
    ref_chosen = sequence_logprob(ref_chosen_logits, chosen_tokens)
    ref_rejected = sequence_logprob(ref_rejected_logits, rejected_tokens)

    dpo, dpo_logits = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, args.beta)
    ipo = ipo_loss(dpo_logits, args.beta)

    chosen_scores = torch.tensor([1.8, 1.2, 0.6, 1.5])
    rejected_scores = torch.tensor([0.2, 0.8, 0.4, 0.1])
    rm_loss = reward_model_pair_loss(chosen_scores, rejected_scores)

    rewards_for_one_prompt = torch.tensor([0.1, 0.4, 1.2, -0.2, 0.8, 0.3])
    advantages = grpo_advantages(rewards_for_one_prompt)

    print("Preference optimization demo")
    print(f"beta={args.beta}")
    print(f"policy_margin_mean={(policy_chosen - policy_rejected).mean().item():.6f}")
    print(f"ref_margin_mean={(ref_chosen - ref_rejected).mean().item():.6f}")
    print(f"dpo_loss={dpo.item():.6f}")
    print(f"ipo_loss={ipo.item():.6f}")
    print(f"reward_model_pair_loss={rm_loss.item():.6f}")
    print(f"grpo_rewards={','.join(f'{value:.2f}' for value in rewards_for_one_prompt.tolist())}")
    print(f"grpo_advantages={','.join(f'{value:.2f}' for value in advantages.tolist())}")


if __name__ == "__main__":
    main()
