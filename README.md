# Reinforcement Learning Sandbox for Atari Breakout


This is a reinforcement learning benchmarking platform I built to study how well RL algorithms generalize and how different reward structures affect learning in Atari Breakout. I needed a way to create tons of Breakout variations and test algorithms under controlled conditions for a research paper.
Since this work is part of a **collaborative paper currently under review**, the code is not publicly available at this time. This document showcases my work.

---

## Motivation

The goal of this project was to create a system capable of:  

- Programmatically generating different Breakout environments (e.g., sparse vs. dense rewards, noisy paddle controls, altered ball physics, visual occlusions).  
- Running large-scale hyperparameter sweeps across multiple RL algorithms.  
- Feeding experimental results directly into a research publication.  

The stack includes **PyTorch** for policies, **Gymnasium** for environments, **Docker** for containerization, and **Nebius AI Studio** for distributed GPU training. Experiments are tracked with **Weights & Biases (W&B)**, and hyperparameters are optimized using **Optuna**.

---

## Architecture
<img width="1536" height="1024" alt="RL Sandbox Overview" src="https://github.com/user-attachments/assets/d8910da3-1c15-4773-b28f-2a8bcddb481b" />

The project is built around an Environment Generator that modifies Breakout’s rules using NumPy while wrapping everything in Gymnasium’s interface. This allows control over brick layouts, ball friction, reward functions, and more. On top of that sit Observation Wrappers that manipulate what the agent perceives—grayscale conversion, random occlusions, frame skipping, and low-resolution encoders—separating visual difficulty from game mechanics.

Implements RL variants of agents, PPO, A2C, Rainbow DQN, in PyTorch, all inheriting from a shared base class for uniform benchmarking. Off-policy algorithms leverage a Prioritized Replay Buffer with a SumTree structure for efficient n-step returns and importance sampling.

Training infrastructure coordinates Docker containers, GPU provisioning via Nebius, experiment tracking with Weights & Biases, and Optuna-driven hyperparameter search. All models are checkpointed to S3.

An Evaluation Harness executes deterministic rollouts on held-out seeds, producing aggregate statistics and CSV logs.

---

## Workflow

Everything is defined by JSON specs that describe brick configs, ball velocity, paddle friction, and reward components. I sample from these to create 20+ environment families. Every environment uses deterministic seeding unless you explicitly enable stochastic wrappers. I kept network architectures separate from training logic. 

Rainbow DQN gets NoisyNets, multi-step returns, and distributional outputs. 
PPO and A2C use shared actor-critic networks with GAE and clipped objectives. Hyperparameters are mostly fixed except when Optuna is tuning them. Built for speed with SumTree-based prioritized sampling. 

N-step returns get computed at insertion time so sampling stays fast.Single-trial mode for debugging, distributed mode for serious experiments. Off-policy algorithms use centralized replay with async actor-learners. On-policy ones collect rollouts and do mini-batch SGD. Everything logs to W&B continuously.

Each experiment starts with a JSON spec defining the environment, observation wrappers, algorithm, and hyperparameters (or Optuna study config). The orchestrator validates this, creates a W&B run, and if it's distributed, tells Nebius to spin up N worker containers. Workers build environments from the spec with deterministic seeding. For off-policy algorithms, actors push transitions to centralized replay while learners sample batches and update networks. For on-policy, they collect rollouts, compute advantages with GAE, and do several epochs of local SGD. Checkpoints save every N steps and upload to S3. When training finishes, the evaluation harness runs deterministic rollouts on held-out seeds and environment variants.

---

##

Since this work feeds directly into a paper, everything is tracked: full specs, random seeds, Docker tags, commit hashes, W&B URLs, and job IDs. Statistical tests report exact p-values and effect sizes, with bootstrapped confidence intervals where appropriate. All provenance data is archived with the manuscript for reviewers.  

---


**Tanmay Shah** — [tanmay.shah@uwaterloo.ca](mailto:tanmay.shah@uwaterloo.ca)
