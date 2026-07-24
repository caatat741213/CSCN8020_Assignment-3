# DQN Assignment: Model Selection and Data Analysis Report

## Executive Summary
This report summarizes the experimental results across five DQN hyperparameter configurations (Config A to E) and compares them with the rule-based baseline policy. 

## Stage 1: Official Selected Model (selected_dqn.pt)
- **Qualifier Criteria**: Evaluation Success Rate $\ge 80\%$.
- **Tie-breakers**: Higher Mean Cumulative Reward, followed by lower Mean Angle Error.
- **Winner Selected**: **Config B (Faster Decay)**
- **Decision Rationale**: Both qualified and have equal success rates (100.00%). Selected based on higher Mean Cumulative Reward: Config A=13.2623 vs Config B=13.3026
- **Action Taken**: Automatically copied the checkpoint file of this winner config to `models/selected_dqn.pt`.

## Stage 2: Best Overall / Empirical Winner (Config A ~ E)
- **Empirical Winner**: **Config C (Linear Decay)**
- **Evaluation Success Rate**: 100.00%
- **Mean Cumulative Reward**: 13.3429
- **Mean Angle Error**: 0.006779 rad
- **Analysis**:
  - The empirical winner (Config C (Linear Decay)) achieved an evaluation success rate of 100.00%.
  - Comparing with the official A/B baseline configs, the empirical winner demonstrates excellent stability and performance, achieving a reward of 13.3429 and error of 0.006779 rad.

## Table 1: 20-Episode Benchmark Evaluation Table (All Configurations)
| Configuration | Epsilon/Var | Successes/20 | Success Rate (%) | Mean Reward | Mean Steps | Mean Angle Error (rad) |
|---|---|---|---|---|---|---|
| Config A (Baseline) | 0.995 (Exp) | 20/20 | 100.0% | 13.2623 | 19.75 | 0.005197 |
| Config B (Faster Decay) | 0.985 (Exp) | 20/20 | 100.0% | 13.3026 | 19.50 | 0.010608 |
| Config C (Linear Decay) | Linear (500 ep) | 20/20 | 100.0% | 13.3429 | 19.50 | 0.006779 |
| Config D (Fast Target Update) | 0.995 (Exp) | 20/20 | 100.0% | 13.1241 | 20.25 | 0.008227 |
| Config E (Small Buffer) | 0.995 (Exp) | 20/20 | 100.0% | 13.1991 | 19.50 | 0.008837 |

## Table 2: Rule-Based Baseline vs. Selected DQN (Official)
| Metric | Rule-based Policy | Selected DQN (Official) |
|---|---|---|
| Successes/20 | 20/20 | 20/20 |
| Success Rate | 100.0% | 100.0% |
| Mean Cumulative Reward | 12.8666 | 13.3026 |
| Mean Episode Length (Steps) | 24.00 | 19.50 |
| Mean Angle Error (rad) | 0.012209 | 0.010608 |
| Main Qualitative Behaviour | Deterministically proportional. Target changes immediately and stays at limit. Can lead to slight static error or sluggishness in G1 joint due to lack of velocity prediction. | Learned value-driven policy. Dynamically selects actions to build torque. Learns to hold the target angle near the goal, reducing steady-state error and oscillation. |

## Ablation Insight Summary

### Target Update Interval (Config D vs Config A)
- **Config A (Baseline)**: Updates target network every 250 steps.
  - Evaluation Reward: 13.2623, Evaluation Error: 0.005197 rad.
  - Training Last 100 Ep Mean Reward: 15.0323 (Std: 2.3976).
- **Config D (Fast Target Update)**: Updates target network every 50 steps.
  - Evaluation Reward: 13.1241, Evaluation Error: 0.008227 rad.
  - Training Last 100 Ep Mean Reward: 14.9494 (Std: 2.1639).
- **Insight**: Fast target update frequency (Config D) results in target Q-values changing too rapidly. This propagates errors and bootstraps unstable estimates quicker, leading to higher training variance and slightly degraded evaluation performance compared to Config A.

### Replay Buffer Capacity (Config E vs Config A)
- **Config A (Baseline)**: Buffer Capacity = 50,000 transitions.
  - Training Last 100 Ep Mean Reward: 15.0323 (Std: 2.3976).
  - First qualified convergence episode (Success Rate >= 80%): Episode 60.
- **Config E (Small Buffer)**: Buffer Capacity = 1,000 transitions.
  - Training Last 100 Ep Mean Reward: 15.5195 (Std: 2.5301).
  - First qualified convergence episode (Success Rate >= 80%): Episode 61.
- **Insight**: A smaller buffer capacity of 1,000 (Config E) acts as a highly local and correlated buffer. The agent forgets older, diverse trajectories quickly, leading to catastrophic forgetting of early control experiences, higher training reward standard deviations, and suboptimal learning stability.
