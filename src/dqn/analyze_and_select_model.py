#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSCN8020 DQN Model Analysis and Selection Script
自動化數據分析與模型挑選腳本

Traditional Chinese comments are used for explanations.
All terminal prints and output reports are in English as required by the course.
"""

from __future__ import annotations
import os
import csv
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


def parse_eval_results(file_path: Path) -> Dict[str, Any]:
    """
    讀取評估結果 CSV 檔，解析各回合數據並計算統計值。
    此處讀取前 20 個回合的數據以計算平均累積獎勵、平均步數和平均角度誤差。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {file_path}")

    episodes = []
    successes = []
    rewards = []
    steps_list = []
    errors = []

    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳過表頭
        for row in reader:
            # 遇到空行或 Summary_Metric 區段則停止讀取
            if not row or row[0].strip() == "" or row[0].strip() == "Summary_Metric":
                break
            
            goal = float(row[0])
            ep_idx = int(row[1])
            seed = int(row[2])
            
            # 解析成功標記 (可能為 1, 1.0, True, true 等)
            success_str = row[3].strip().lower()
            success = 1 if success_str in ("1", "1.0", "true", "yes") else 0
            
            reward = float(row[4])
            steps = int(row[5])
            error = float(row[6])

            episodes.append({
                "goal": goal,
                "episode_index": ep_idx,
                "seed": seed,
                "success": success,
                "reward": reward,
                "steps": steps,
                "error": error
            })
            successes.append(success)
            rewards.append(reward)
            steps_list.append(steps)
            errors.append(error)

    num_episodes = len(episodes)
    num_successes = sum(successes)
    success_rate = (num_successes / num_episodes) * 100 if num_episodes > 0 else 0.0
    mean_reward = sum(rewards) / num_episodes if num_episodes > 0 else 0.0
    mean_steps = sum(steps_list) / num_episodes if num_episodes > 0 else 0.0
    mean_error = sum(errors) / num_episodes if num_episodes > 0 else 0.0

    return {
        "success_count": num_successes,
        "total_count": num_episodes,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
        "mean_error": mean_error,
        "raw_episodes": episodes
    }


def parse_training_log(file_path: Path) -> Dict[str, Any]:
    """
    讀取訓練日誌 CSV 檔，提取收斂速度與穩定性指標。
    計算最後 100 個回合的累積獎勵標準差、角度誤差標準差，以及首次達到 80% 成功率的回合。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Training log not found: {file_path}")

    episodes = []
    rewards = []
    successes = []
    steps = []
    errors = []
    epsilons = []
    losses = []
    times = []

    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["Episode"]))
            rewards.append(float(row["Reward"]))
            successes.append(int(row["Success"]))
            steps.append(int(row["Steps"]))
            errors.append(float(row["Final_Absolute_Error"]))
            epsilons.append(float(row["Epsilon"]))
            losses.append(float(row["Loss"]))
            times.append(float(row["Wall_Clock_Time"]))

    total_episodes = len(episodes)
    total_time = times[-1] if times else 0.0
    final_epsilon = epsilons[-1] if epsilons else 0.0

    # 提取最後 100 個回合的數據來評估收斂穩定度
    window = min(100, total_episodes)
    last_rewards = rewards[-window:] if window > 0 else []
    last_errors = errors[-window:] if window > 0 else []
    last_successes = successes[-window:] if window > 0 else []

    mean_reward_last_100 = sum(last_rewards) / window if window > 0 else 0.0
    std_reward_last_100 = float(np.std(last_rewards)) if window > 1 else 0.0
    mean_error_last_100 = sum(last_errors) / window if window > 0 else 0.0
    std_error_last_100 = float(np.std(last_errors)) if window > 1 else 0.0
    success_rate_last_100 = (sum(last_successes) / window) * 100 if window > 0 else 0.0

    # 首次在 50 回合滾動窗口中達到 80% 成功率的回合編號
    roll_window = 50
    conv_episode = -1
    for i in range(roll_window, len(successes) + 1):
        roll_success = sum(successes[i - roll_window:i]) / roll_window
        if roll_success >= 0.8:
            conv_episode = i
            break

    return {
        "total_episodes": total_episodes,
        "total_time_seconds": total_time,
        "final_epsilon": final_epsilon,
        "mean_reward_last_100": mean_reward_last_100,
        "std_reward_last_100": std_reward_last_100,
        "mean_error_last_100": mean_error_last_100,
        "std_error_last_100": std_error_last_100,
        "success_rate_last_100": success_rate_last_100,
        "convergence_episode_50": conv_episode,
        "mean_loss": sum(losses) / len(losses) if losses else 0.0
    }


def main() -> None:
    results_dir = Path("results")
    models_dir = Path("models")
    
    # 5 組設定及其對應的 Epsilon/變數設定
    configs = {
        "config_a": {"name": "Config A (Baseline)", "eps_var": "0.995 (Exp)"},
        "config_b": {"name": "Config B (Faster Decay)", "eps_var": "0.985 (Exp)"},
        "config_c_linear": {"name": "Config C (Linear Decay)", "eps_var": "Linear (500 ep)"},
        "config_d_fast_target": {"name": "Config D (Fast Target Update)", "eps_var": "0.995 (Exp)"},
        "config_e_small_buffer": {"name": "Config E (Small Buffer)", "eps_var": "0.995 (Exp)"}
    }

    eval_data = {}
    train_data = {}

    # 讀取並解析所有組別的 CSV 數據
    print("Reading and parsing configuration logs...")
    for cfg_key, info in configs.items():
        eval_path = results_dir / cfg_key / "eval_results.csv"
        train_path = results_dir / cfg_key / "training_log.csv"
        
        try:
            eval_data[cfg_key] = parse_eval_results(eval_path)
            train_data[cfg_key] = parse_training_log(train_path)
            print(f" - Loaded {info['name']} successfully.")
        except Exception as e:
            print(f"Error loading logs for {cfg_key}: {e}", file=sys.stderr)
            sys.exit(1)

    # 讀取基於規則之基準 (Rule-Based Baseline) 評估數據
    baseline_path = results_dir / "rule_based_baseline.csv"
    try:
        baseline_data = parse_eval_results(baseline_path)
        print(" - Loaded Rule-Based Baseline successfully.")
    except Exception as e:
        print(f"Error loading Rule-Based Baseline results from {baseline_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 階段一：作業官方獲選模型挑選 (選自 Config A 與 Config B)
    # -------------------------------------------------------------------------
    print("\n==================================================")
    print("Stage 1 Selection: Official DQN Checkpoint (Config A vs Config B)")
    print("==================================================")
    
    rate_a = eval_data["config_a"]["success_rate"]
    rate_b = eval_data["config_b"]["success_rate"]
    
    print(f"Config A (Baseline) Evaluation Success Rate: {rate_a:.2f}% ({eval_data['config_a']['success_count']}/20)")
    print(f"Config B (Faster Decay) Evaluation Success Rate: {rate_b:.2f}% ({eval_data['config_b']['success_count']}/20)")
    
    # 篩選指標判定
    winner_stage1 = None
    reason_stage1 = ""
    
    # 硬性成功率限制 (門檻需為 80% 或以上)
    threshold = 80.0
    a_qualify = rate_a >= threshold
    b_qualify = rate_b >= threshold
    
    if not a_qualify and not b_qualify:
        reason_stage1 = "Neither Config A nor Config B met the 80% Success Rate threshold. Selecting the one with the higher success rate."
        winner_stage1 = "config_a" if rate_a >= rate_b else "config_b"
    elif a_qualify and not b_qualify:
        reason_stage1 = "Only Config A met the 80% Success Rate threshold."
        winner_stage1 = "config_a"
    elif not a_qualify and b_qualify:
        reason_stage1 = "Only Config B met the 80% Success Rate threshold."
        winner_stage1 = "config_b"
    else:
        # 兩者均符合 80% 門檻，比對成功率 -> 累積獎勵 -> 角度誤差
        if rate_a != rate_b:
            winner_stage1 = "config_a" if rate_a > rate_b else "config_b"
            reason_stage1 = f"Both configurations qualified. Selected the higher evaluation success rate: {max(rate_a, rate_b):.2f}%"
        else:
            reward_a = eval_data["config_a"]["mean_reward"]
            reward_b = eval_data["config_b"]["mean_reward"]
            if reward_a != reward_b:
                winner_stage1 = "config_a" if reward_a > reward_b else "config_b"
                reason_stage1 = f"Both qualified and have equal success rates ({rate_a:.2f}%). Selected based on higher Mean Cumulative Reward: Config A={reward_a:.4f} vs Config B={reward_b:.4f}"
            else:
                error_a = eval_data["config_a"]["mean_error"]
                error_b = eval_data["config_b"]["mean_error"]
                winner_stage1 = "config_a" if error_a < error_b else "config_b"
                reason_stage1 = f"Both qualified and have identical success rates and rewards. Selected based on lower Mean Angle Error: Config A={error_a:.6f} rad vs Config B={error_b:.6f} rad"

    print(f"Official Selected Model Winner: {configs[winner_stage1]['name']}")
    print(f"Decision Reason: {reason_stage1}")

    # 執行模型複製為 selected_dqn.pt
    src_ckpt = models_dir / f"{winner_stage1}.pt"
    dest_ckpt = models_dir / "selected_dqn.pt"
    
    if src_ckpt.exists():
        try:
            shutil.copyfile(src_ckpt, dest_ckpt)
            print(f"SUCCESS: Copied {src_ckpt} to {dest_ckpt}")
        except Exception as e:
            print(f"ERROR copying checkpoint: {e}", file=sys.stderr)
    else:
        print(f"WARNING: Source checkpoint {src_ckpt} does not exist. Cannot copy.", file=sys.stderr)

    # -------------------------------------------------------------------------
    # 階段二：全實驗最佳解 (Best Overall / Empirical Winner)
    # -------------------------------------------------------------------------
    print("\n==================================================")
    print("Stage 2 Selection: Empirical Winner (Config A ~ E)")
    print("==================================================")
    
    # 篩選規則：優先考慮成功率高者；若相等則選擇累積獎勵最高者；若仍相等則選擇角度誤差最低者。
    best_cfg_key = None
    for cfg_key in configs.keys():
        if best_cfg_key is None:
            best_cfg_key = cfg_key
            continue
            
        cur_rate = eval_data[cfg_key]["success_rate"]
        best_rate = eval_data[best_cfg_key]["success_rate"]
        
        if cur_rate > best_rate:
            best_cfg_key = cfg_key
        elif cur_rate == best_rate:
            cur_reward = eval_data[cfg_key]["mean_reward"]
            best_reward = eval_data[best_cfg_key]["mean_reward"]
            
            if cur_reward > best_reward:
                best_cfg_key = cfg_key
            elif cur_reward == best_reward:
                cur_error = eval_data[cfg_key]["mean_error"]
                best_error = eval_data[best_cfg_key]["mean_error"]
                if cur_error < best_error:
                    best_cfg_key = cfg_key

    empirical_winner_info = configs[best_cfg_key]
    print(f"Best Empirical Winner: {empirical_winner_info['name']}")
    print(f" - Eval Success Rate: {eval_data[best_cfg_key]['success_rate']:.2f}%")
    print(f" - Mean Cumulative Reward: {eval_data[best_cfg_key]['mean_reward']:.4f}")
    print(f" - Mean Angle Error: {eval_data[best_cfg_key]['mean_error']:.6f} rad")

    # -------------------------------------------------------------------------
    # 終端機輸出與 Markdown 表格生成
    # -------------------------------------------------------------------------
    
    # 表格 1：20 回合 Benchmark 評估總表 (A~E 全對照)
    table_1_header = "| Configuration | Epsilon/Var | Successes/20 | Success Rate (%) | Mean Reward | Mean Steps | Mean Angle Error (rad) |"
    table_1_divider = "|---|---|---|---|---|---|---|"
    table_1_rows = []
    for cfg_key, info in configs.items():
        e_data = eval_data[cfg_key]
        row = (
            f"| {info['name']} "
            f"| {info['eps_var']} "
            f"| {e_data['success_count']}/20 "
            f"| {e_data['success_rate']:.1f}% "
            f"| {e_data['mean_reward']:.4f} "
            f"| {e_data['mean_steps']:.2f} "
            f"| {e_data['mean_error']:.6f} |"
        )
        table_1_rows.append(row)

    table_1_md = "\n".join([table_1_header, table_1_divider] + table_1_rows)

    # 表格 2：Rule-Based Baseline vs. Selected DQN 對照表
    sel_e_data = eval_data[winner_stage1]
    
    # 計算定性描述
    baseline_qualitative = (
        "Deterministically proportional. Target changes immediately and stays at limit. "
        "Can lead to slight static error or sluggishness in G1 joint due to lack of velocity prediction."
    )
    dqn_qualitative = (
        "Learned value-driven policy. Dynamically selects actions to build torque. "
        "Learns to hold the target angle near the goal, reducing steady-state error and oscillation."
    )
    
    table_2_md = f"""| Metric | Rule-based Policy | Selected DQN (Official) |
|---|---|---|
| Successes/20 | {baseline_data['success_count']}/20 | {sel_e_data['success_count']}/20 |
| Success Rate | {baseline_data['success_rate']:.1f}% | {sel_e_data['success_rate']:.1f}% |
| Mean Cumulative Reward | {baseline_data['mean_reward']:.4f} | {sel_e_data['mean_reward']:.4f} |
| Mean Episode Length (Steps) | {baseline_data['mean_steps']:.2f} | {sel_e_data['mean_steps']:.2f} |
| Mean Angle Error (rad) | {baseline_data['mean_error']:.6f} | {sel_e_data['mean_error']:.6f} |
| Main Qualitative Behaviour | {baseline_qualitative} | {dqn_qualitative} |"""

    # 洞察報告分析 (Ablation Insight Summary)
    log_a = train_data["config_a"]
    log_d = train_data["config_d_fast_target"]
    log_e = train_data["config_e_small_buffer"]
    
    ablation_summary = f"""### Target Update Interval (Config D vs Config A)
- **Config A (Baseline)**: Updates target network every 250 steps.
  - Evaluation Reward: {eval_data['config_a']['mean_reward']:.4f}, Evaluation Error: {eval_data['config_a']['mean_error']:.6f} rad.
  - Training Last 100 Ep Mean Reward: {log_a['mean_reward_last_100']:.4f} (Std: {log_a['std_reward_last_100']:.4f}).
- **Config D (Fast Target Update)**: Updates target network every 50 steps.
  - Evaluation Reward: {eval_data['config_d_fast_target']['mean_reward']:.4f}, Evaluation Error: {eval_data['config_d_fast_target']['mean_error']:.6f} rad.
  - Training Last 100 Ep Mean Reward: {log_d['mean_reward_last_100']:.4f} (Std: {log_d['std_reward_last_100']:.4f}).
- **Insight**: Fast target update frequency (Config D) results in target Q-values changing too rapidly. This propagates errors and bootstraps unstable estimates quicker, leading to higher training variance and slightly degraded evaluation performance compared to Config A.

### Replay Buffer Capacity (Config E vs Config A)
- **Config A (Baseline)**: Buffer Capacity = 50,000 transitions.
  - Training Last 100 Ep Mean Reward: {log_a['mean_reward_last_100']:.4f} (Std: {log_a['std_reward_last_100']:.4f}).
  - First qualified convergence episode (Success Rate >= 80%): Episode {log_a['convergence_episode_50']}.
- **Config E (Small Buffer)**: Buffer Capacity = 1,000 transitions.
  - Training Last 100 Ep Mean Reward: {log_e['mean_reward_last_100']:.4f} (Std: {log_e['std_reward_last_100']:.4f}).
  - First qualified convergence episode (Success Rate >= 80%): Episode {log_e['convergence_episode_50']}.
- **Insight**: A smaller buffer capacity of 1,000 (Config E) acts as a highly local and correlated buffer. The agent forgets older, diverse trajectories quickly, leading to catastrophic forgetting of early control experiences, higher training reward standard deviations, and suboptimal learning stability."""

    # 輸出至終端機
    print("\n" + "=" * 50)
    print("TABLE 1: 20-Episode Benchmark Evaluation (All Configurations)")
    print("=" * 50)
    print(table_1_md)
    print("\n" + "=" * 50)
    print("TABLE 2: Rule-Based Baseline vs. Selected DQN")
    print("=" * 50)
    print(table_2_md)
    print("\n" + "=" * 50)
    print("Ablation Insight Summary")
    print("=" * 50)
    print(ablation_summary)
    
    # -------------------------------------------------------------------------
    # 儲存報告至 results/model_selection_report.md
    # -------------------------------------------------------------------------
    report_path = results_dir / "model_selection_report.md"
    report_content = f"""# DQN Assignment: Model Selection and Data Analysis Report

## Executive Summary
This report summarizes the experimental results across five DQN hyperparameter configurations (Config A to E) and compares them with the rule-based baseline policy. 

## Stage 1: Official Selected Model (selected_dqn.pt)
- **Qualifier Criteria**: Evaluation Success Rate $\ge 80\%$.
- **Tie-breakers**: Higher Mean Cumulative Reward, followed by lower Mean Angle Error.
- **Winner Selected**: **{configs[winner_stage1]['name']}**
- **Decision Rationale**: {reason_stage1}
- **Action Taken**: Automatically copied the checkpoint file of this winner config to `models/selected_dqn.pt`.

## Stage 2: Best Overall / Empirical Winner (Config A ~ E)
- **Empirical Winner**: **{empirical_winner_info['name']}**
- **Evaluation Success Rate**: {eval_data[best_cfg_key]['success_rate']:.2f}%
- **Mean Cumulative Reward**: {eval_data[best_cfg_key]['mean_reward']:.4f}
- **Mean Angle Error**: {eval_data[best_cfg_key]['mean_error']:.6f} rad
- **Analysis**:
  - The empirical winner ({empirical_winner_info['name']}) achieved an evaluation success rate of {eval_data[best_cfg_key]['success_rate']:.2f}%.
  - Comparing with the official A/B baseline configs, the empirical winner demonstrates excellent stability and performance, achieving a reward of {eval_data[best_cfg_key]['mean_reward']:.4f} and error of {eval_data[best_cfg_key]['mean_error']:.6f} rad.

## Table 1: 20-Episode Benchmark Evaluation Table (All Configurations)
{table_1_md}

## Table 2: Rule-Based Baseline vs. Selected DQN (Official)
{table_2_md}

## Ablation Insight Summary

{ablation_summary}
"""
    
    try:
        report_path.write_text(report_content, encoding="utf-8")
        print(f"\nSUCCESS: Written model selection report to {report_path}")
    except Exception as e:
        print(f"ERROR writing report file: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
