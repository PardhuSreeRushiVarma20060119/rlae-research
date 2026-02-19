# 🤖 Models Directory

This directory contains the persisted behavioral artifacts (LoRA adapters) generated during the research lifecycle.

## Subdirectories
- **`lora_sft/`**: The adapter generated during the Supervised Fine-Tuning phase. Represents the "Instruction-Aligned" environment.
- **`lora_rl/`**: The adapter generated during the Reinforcement Learning (DPO) phase. Represents the "Preference-Aligned" environment.

## Canonical Principle
In accordance with **RLAE (Runtime Low-Rank Adaptive Environments)**, these adapters are the *only* parts of the system that change. The base model remains frozen and is never stored here.
