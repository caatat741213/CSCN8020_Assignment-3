"""
Plotting Script for DQN training and evaluation results.
This script reads the CSV training and evaluation log files for all 5 configurations:
Config A (Baseline), Config B (Faster Decay), Config C (Linear Decay),
Config D (Fast Target Update), and Config E (Small Buffer).
It plots the five required comparative charts and saves them to results/plots/ (300 DPI).

DQN 訓練與評估結果繪圖與對照腳本。
讀取 Config A, B, C, D, E 的 CSV 檔，繪製五張高解析度 (300 DPI) 比較圖表。
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Try to use a clean modern style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    try:
        plt.style.use("ggplot")
    except:
        pass


def read_training_log(csv_path: Path) -> dict[str, np.ndarray] | None:
    """
    Read training log CSV and return as a dict of numpy arrays.
    讀取訓練日誌 CSV 檔並回傳 numpy 陣列字典。
    """
    if not csv_path.exists():
        print(f"Warning: Training log not found at {csv_path}")
        return None
        
    episodes = []
    rewards = []
    successes = []
    steps = []
    errors = []
    epsilons = []
    losses = []
    times = []
    
    with open(csv_path, mode="r", encoding="utf-8") as f:
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
            
    return {
        "Episode": np.array(episodes),
        "Reward": np.array(rewards),
        "Success": np.array(successes),
        "Steps": np.array(steps),
        "Final_Absolute_Error": np.array(errors),
        "Epsilon": np.array(epsilons),
        "Loss": np.array(losses),
        "Wall_Clock_Time": np.array(times),
    }


def read_eval_results(csv_path: Path) -> dict[str, np.ndarray] | None:
    """
    Read evaluation results CSV, parsing only the first 20 data rows.
    讀取評估結果 CSV 檔，僅解析前 20 筆資料列。
    """
    if not csv_path.exists():
        print(f"Warning: Evaluation results not found at {csv_path}")
        return None
        
    goals = []
    episodes = []
    seeds = []
    successes = []
    rewards = []
    lengths = []
    errors = []
    
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            _header = next(reader)  # skip header
        except StopIteration:
            return None
            
        for row in reader:
            # Stop if we hit an empty row or the summary section
            if not row or row[0] == "" or row[0] == "Summary_Metric":
                break
            goals.append(float(row[0]))
            episodes.append(int(row[1]))
            seeds.append(int(row[2]))
            successes.append(int(row[3]))
            rewards.append(float(row[4]))
            lengths.append(int(row[5]))
            errors.append(float(row[6]))
            
    return {
        "Goal": np.array(goals),
        "Episode": np.array(episodes),
        "Seed": np.array(seeds),
        "Success": np.array(successes),
        "Reward": np.array(rewards),
        "Length": np.array(lengths),
        "Final_Absolute_Error": np.array(errors),
    }


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate moving average using convolution. Handles short arrays.
    使用卷積計算滑動平均，包含長度不足時的自適應處理。
    """
    if len(data) == 0:
        return np.array([])
    actual_window = min(window, len(data))
    return np.convolve(data, np.ones(actual_window) / actual_window, mode="valid")


def main() -> None:
    results_dir = Path("results")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    configs = ["config_a", "config_b", "config_c_linear", "config_d_fast_target", "config_e_small_buffer"]
    
    # Load all training logs
    logs = {}
    for cfg in configs:
        logs[cfg] = read_training_log(results_dir / cfg / "training_log.csv")
        
    # Load all evaluation logs
    evals = {}
    for cfg in configs:
        evals[cfg] = read_eval_results(results_dir / cfg / "eval_results.csv")
        
    # Professional Styling Configuration (HSL/Sleek Theme Palette)
    colors = {
        "config_a": "#1f77b4",          # Tech blue (Baseline)
        "config_a_light": "#a6c8e0",
        "config_b": "#ff7f0e",          # Coral orange (Faster Decay)
        "config_b_light": "#ffcfa6",
        "config_c_linear": "#2ca02c",   # Emerald green (Linear Decay)
        "config_c_light": "#abe6ab",
        "config_d_fast_target": "#d62728",  # Crimson red (Fast Target Update)
        "config_d_light": "#f3a5a5",
        "config_e_small_buffer": "#9467bd", # Orchid purple (Small Replay Buffer)
        "config_e_light": "#d8c4eb"
    }
    
    labels = {
        "config_a": "Config A: Baseline (decay=0.995)",
        "config_b": "Config B: Faster Decay (decay=0.985)",
        "config_c_linear": "Config C: Linear Decay",
        "config_d_fast_target": "Config D: Fast Target Update (50 steps)",
        "config_e_small_buffer": "Config E: Small Buffer (capacity=1000)"
    }

    # -------------------------------------------------------------------------
    # Chart a: training_rewards.png (Comparison of Config A and B)
    # -------------------------------------------------------------------------
    print("Generating training_rewards.png...")
    plt.figure(figsize=(11, 6.5))
    
    log_a = logs["config_a"]
    log_b = logs["config_b"]
    
    if log_a is not None:
        plt.plot(log_a["Episode"], log_a["Reward"], color=colors["config_a_light"], alpha=0.3, label="Config A Raw")
        ma_a = moving_average(log_a["Reward"], window=30)
        w_a = min(30, len(log_a["Reward"]))
        if len(ma_a) > 0:
            plt.plot(np.arange(w_a, len(log_a["Reward"]) + 1), ma_a, color=colors["config_a"], linewidth=2.5, label="Config A Moving Avg (w=30)")
            
    if log_b is not None:
        plt.plot(log_b["Episode"], log_b["Reward"], color=colors["config_b_light"], alpha=0.3, label="Config B Raw")
        ma_b = moving_average(log_b["Reward"], window=30)
        w_b = min(30, len(log_b["Reward"]))
        if len(ma_b) > 0:
            plt.plot(np.arange(w_b, len(log_b["Reward"]) + 1), ma_b, color=colors["config_b"], linewidth=2.5, label="Config B Moving Avg (w=30)")
            
    plt.title("DQN Training Cumulative Reward Comparison: Baseline vs Faster Decay", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_rewards.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart b: training_success_rate.png (All configs rolling success rate)
    # -------------------------------------------------------------------------
    print("Generating training_success_rate.png...")
    plt.figure(figsize=(11, 6.5))
    
    for cfg in configs:
        log_cfg = logs[cfg]
        if log_cfg is not None:
            ma_success = moving_average(log_cfg["Success"], window=50) * 100.0
            w_cfg = min(50, len(log_cfg["Success"]))
            if len(ma_success) > 0:
                plt.plot(np.arange(w_cfg, len(log_cfg["Success"]) + 1), ma_success, color=colors[cfg], linewidth=2, label=labels[cfg])
                
    plt.title("Rolling Training Success Rate (window=50) across Configurations", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.ylim(-5, 105)
    plt.legend(loc="lower right", frameon=True, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_success_rate.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart c: epsilon_decay.png (Epsilon Curves Comparison)
    # -------------------------------------------------------------------------
    print("Generating epsilon_decay.png...")
    plt.figure(figsize=(10, 6))
    
    # We compare Exponential (0.995 vs 0.985) and Linear Decay
    for cfg in ["config_a", "config_b", "config_c_linear"]:
        log_cfg = logs[cfg]
        if log_cfg is not None:
            plt.plot(log_cfg["Episode"], log_cfg["Epsilon"], color=colors[cfg], linewidth=2.5, label=labels[cfg])
            
    plt.title("Epsilon (Exploration Rate) Decay Schedule Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Epsilon Value", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "epsilon_decay.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart d: training_loss.png (Loss Fluctuations Comparison)
    # -------------------------------------------------------------------------
    print("Generating training_loss.png...")
    plt.figure(figsize=(11, 6.5))
    
    # Compare baseline (Config A), Fast Target Update (Config D), and Small Buffer (Config E)
    for cfg in ["config_a", "config_d_fast_target", "config_e_small_buffer"]:
        log_cfg = logs[cfg]
        if log_cfg is not None:
            # Filter out episodes where optimization didn't run (loss == 0)
            valid_mask = log_cfg["Loss"] > 0
            episodes_valid = log_cfg["Episode"][valid_mask]
            losses_valid = log_cfg["Loss"][valid_mask]
            
            if len(losses_valid) > 0:
                # Plot raw lightly
                plt.plot(episodes_valid, losses_valid, color=colors[f"{cfg}_light"], alpha=0.25)
                # Plot smoothed
                ma_loss = moving_average(losses_valid, window=30)
                w_cfg = min(30, len(losses_valid))
                if len(ma_loss) > 0:
                    plt.plot(episodes_valid[w_cfg - 1:], ma_loss, color=colors[cfg], linewidth=2, label=labels[cfg])
                    
    plt.title("Huber Loss Fluctuations: Effect of Target Update Speed and Replay Buffer Size", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Huber Loss", fontsize=12)
    plt.yscale("log")  # Use log scale due to large spikes
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_loss.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # Chart e: eval_success_by_angle.png (Greedy Success Rates by Angle)
    # -------------------------------------------------------------------------
    print("Generating eval_success_by_angle.png...")
    
    goals = [-0.8, -0.4, 0.4, 0.8]
    x = np.arange(len(goals))
    width = 0.15  # Adjust width to fit 5 configs
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Shift position of bars for each configuration
    offsets = [-2 * width, -1 * width, 0.0, 1 * width, 2 * width]
    
    for idx, cfg in enumerate(configs):
        eval_cfg = evals[cfg]
        success_rates = []
        
        for g in goals:
            if eval_cfg is not None:
                match_indices = np.where(np.isclose(eval_cfg["Goal"], g))[0]
                if len(match_indices) > 0:
                    rate = np.mean(eval_cfg["Success"][match_indices]) * 100.0
                else:
                    rate = 0.0
            else:
                rate = 0.0
            success_rates.append(rate)
            
        rects = ax.bar(x + offsets[idx], success_rates, width, label=labels[cfg], color=colors[cfg])
        
        # Add values on top of bars
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.0f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 2 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold"
                )
                
    ax.set_ylabel("Evaluation Success Rate (%)", fontsize=12)
    ax.set_xlabel("Target Angle (rad)", fontsize=12)
    ax.set_title("Greedy Evaluation Success Rate by Target Angle (5 Episodes per Angle)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g:+.1f} rad" for g in goals], fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "eval_success_by_angle.png", dpi=300)
    plt.close()
    
    print("==================================================")
    print("All charts successfully generated and saved to results/plots/")
    print("==================================================")


if __name__ == "__main__":
    main()
