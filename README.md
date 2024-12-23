# Language Imitation via IRL

This repository contains an implementation attempt of the paper <span style="color: #0366d6">["Imitating Language via Scalable Inverse Reinforcement Learning"](https://arxiv.org/pdf/2409.01369)</span>.

## Overview

This is an experimental codebase that aims to implement the language imitation approach described in the paper. The implementation uses inverse reinforcement learning (IRL) techniques to learn language models by inferring rewards from expert demonstrations.

## Acknowledgments

The supervised fine-tuning (SFT) code is adapted from <span style="color: #0366d6">[summarize_from_feedback_details](https://github.com/vwxyzjn/summarize_from_feedback_details)</span> repository.

## Key Components

- Implementation of soft Q-learning for language models
- Reward inference using inverse reinforcement learning
- Integration with supervised fine-tuning framework
- Training and evaluation utilities

## Implementation Details

The codebase includes:
- Custom loss functions for IRL training
- Gradient clipping and optimization utilities
- Training loop implementations
- Evaluation metrics and logging

## Setup and Usage

[Instructions for setup and usage will be added as the implementation progresses]

## Notes

This is an experimental implementation and may not fully replicate all aspects of the original paper. Use with caution and expect ongoing changes and improvements.

## Citation
@code{
  author = {sankyde},
  url    = {https://github.com/sankyde/IRL},
  year   = {2024}
}