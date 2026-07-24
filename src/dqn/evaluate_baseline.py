"""
Verification script for Gymnasium environment and Rule-based Policy.
This script tests the rule-based policy on 4 benchmark goals, each for 5 episodes.
It prints key metrics and saves them to results/rule_based_baseline.csv.

自動驗證 Gymnasium 環境與測試基線（Rule-based Policy）的腳本。
本腳本在 4 個基準目標角度上測試基線策略，每個目標測試 5 個回合。
它會印出關鍵指標並將其儲存至 results/rule_based_baseline.csv 檔案中。
"""

from __future__ import annotations
import os
import csv
import sys
import random
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

# Add 'src' directory to Python path so we can import g1_rl and test_g1_elbow_env
# 將 'src' 目錄加入 Python 路徑，以便匯入 g1_rl 與 test_g1_elbow_env
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from g1_rl import G1ElbowTargetEnv
    from test_g1_elbow_env import choose_rule_based_action
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure you are running this script from the project root directory.")
    sys.exit(1)

# Try importing torch to set seeds if available, but do not fail if not installed
# 嘗試匯入 torch 以設定隨機種子，但如果尚未安裝則不報錯
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    設定隨機種子以確保實驗的可重複性。
    """
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)


def run_evaluation() -> None:
    """
    Run evaluation on the 4 benchmark goals and record metrics.
    在 4 個基準目標上執行評估並記錄指標。
    """
    # Create the environment in headless mode (no viewer rendering)
    # 以無介面模式（不開啟 viewer 畫面）建立 Gymnasium 環境
    env = G1ElbowTargetEnv(render_mode=None)

    # Validate the environment using Gymnasium checker first
    # 首先使用 Gymnasium 檢查器來驗證環境結構
    print("==================================================")
    print("Step 1: Running Gymnasium environment checker...")
    print("==================================================")
    try:
        check_env(env, skip_render_check=True)
        print("Environment checker passed successfully!\n")
    except Exception as e:
        print(f"Environment checker failed: {e}")
        env.close()
        sys.exit(1)

    # Benchmark target angles as specified in the assignment
    # 作業中指定的 4 個基準目標角度（以弧度為單位）
    benchmark_goals = [-0.8, -0.4, 0.4, 0.8]
    episodes_per_goal = 5
    base_seed = 666

    # Lists to store episode metrics
    # 用於儲存各回合指標的列表
    episode_results = []
    
    # Cumulative stats
    # 累計的數據統計
    cumulative_rewards = []
    episode_lengths = []
    final_absolute_errors = []
    success_count = 0

    print("==================================================")
    print("Step 2: Starting evaluation on 4 benchmark goals...")
    print("Testing each goal for 5 episodes (20 episodes total)")
    print("==================================================")
    print(f"{'Goal (rad)':<12} | {'Episode':<8} | {'Seed':<6} | {'Success':<8} | {'Reward':<10} | {'Steps':<6} | {'Final Error':<12}")
    print("-" * 80)

    # Evaluate the rule-based policy
    # 評估基線的規則策略
    for goal in benchmark_goals:
        for ep in range(episodes_per_goal):
            episode_seed = base_seed + ep
            set_seed(episode_seed)

            # Reset the environment with the specific target goal
            # 使用特定的目標角度和種子重置環境
            # We pass the target angle in 'options' as supported by G1ElbowTargetEnv.reset()
            observation, info = env.reset(seed=episode_seed, options={"goal_angle": goal})

            ep_reward = 0.0
            steps = 0

            while True:
                # Choose action using the rule-based baseline policy
                # 使用專案內建的規則型策略選擇動作
                action = choose_rule_based_action(
                    observation=observation,
                    controller_target=float(info["controller_target"]),
                    action_increment=env.action_increment,
                )

                # Step the environment
                # 執行一步模擬環境
                observation, reward, terminated, truncated, info = env.step(action)

                ep_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            # Get final outcome
            # 取得該回合的最終結果
            is_success = info.get("is_success", False)
            final_err = info.get("absolute_error", abs(info["goal_angle"] - info["elbow_angle"]))

            # Record stats
            # 記錄統計數據
            if is_success:
                success_count += 1
            
            cumulative_rewards.append(ep_reward)
            episode_lengths.append(steps)
            final_absolute_errors.append(final_err)

            # Print current episode results to console
            # 將目前回合的結果印出至主控台
            print(f"{goal:<12.1f} | {ep+1:<8d} | {episode_seed:<6d} | {str(is_success):<8} | {ep_reward:<10.4f} | {steps:<6d} | {final_err:<12.6f}")

            # Append to detailed results
            # 附加至詳細結果清單中
            episode_results.append({
                "Goal": goal,
                "Episode": ep + 1,
                "Seed": episode_seed,
                "Success": is_success,
                "Cumulative_Reward": ep_reward,
                "Episode_Length": steps,
                "Final_Absolute_Error": final_err,
            })

    # Close the environment
    # 關閉模擬環境
    env.close()

    # Compute overall metrics
    # 計算整體指標
    total_episodes = len(benchmark_goals) * episodes_per_goal
    success_rate = (success_count / total_episodes) * 100.0
    mean_reward = np.mean(cumulative_rewards)
    mean_length = np.mean(episode_lengths)
    mean_final_error = np.mean(final_absolute_errors)

    # Print summary statistics
    # 印出整體統計摘要
    print("-" * 80)
    print("Evaluation Summary (Rule-based Policy):")
    print(f"Total Successes: {success_count} / {total_episodes}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Mean Cumulative Reward: {mean_reward:.4f}")
    print(f"Mean Episode Length: {mean_length:.2f} steps")
    print(f"Mean Final Absolute Error: {mean_final_error:.6f} rad")
    print("-" * 80)

    # Ensure results directory exists
    # 確保 results 資料夾存在
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "rule_based_baseline.csv"

    # Save details and summary to CSV file
    # 將詳細回合資料和摘要統計存檔至 CSV 檔案
    print(f"Saving baseline results to {csv_path}...")
    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Write detailed run headers
            # 寫入詳細回合的標題列
            writer.writerow([
                "Goal_Angle_rad", 
                "Episode_Index", 
                "Seed", 
                "Success", 
                "Cumulative_Reward", 
                "Episode_Length", 
                "Final_Absolute_Error"
            ])
            
            # Write each episode data row
            # 寫入每一回合的資料行
            for res in episode_results:
                writer.writerow([
                    res["Goal"],
                    res["Episode"],
                    res["Seed"],
                    res["Success"],
                    res["Cumulative_Reward"],
                    res["Episode_Length"],
                    res["Final_Absolute_Error"],
                ])
                
            # Write an empty line to separate detailed runs and summary
            # 寫入空行以區隔詳細資料與統計摘要
            writer.writerow([])
            
            # Write summary metrics
            # 寫入統計摘要
            writer.writerow(["Summary_Metric", "Value"])
            writer.writerow(["Successes", f"{success_count}/{total_episodes}"])
            writer.writerow(["Success_Rate_Percent", f"{success_rate:.2f}"])
            writer.writerow(["Mean_Cumulative_Reward", f"{mean_reward:.4f}"])
            writer.writerow(["Mean_Episode_Length", f"{mean_length:.2f}"])
            writer.writerow(["Mean_Final_Absolute_Error_rad", f"{mean_final_error:.6f}"])
            
        print("Results successfully saved!")
    except Exception as e:
        print(f"Failed to save CSV file: {e}")


if __name__ == "__main__":
    run_evaluation()
