"""
DQN Evaluation Script for Unitree G1 Left Elbow Control.
This script evaluates a trained DQN agent on the 4 benchmark goal angles:
[-0.8, -0.4, 0.4, 0.8] rad, running 5 episodes each (20 episodes total)
with epsilon = 0.0 (greedy policy). It saves detailed results to CSV.

CSCN8020 DQN 評估腳本。
在 4 個基準目標角度上以貪婪策略 (epsilon = 0.0) 評估訓練好的代理人，共 20 回合，並將結果存為 CSV。
"""

from __future__ import annotations
import os
import sys
import csv
import argparse
from pathlib import Path
import numpy as np
import torch

# Add 'src' directory to Python path to allow imports of g1_rl and dqn modules
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from g1_rl import G1ElbowTargetEnv
try:
    from dqn.agent import DQNAgent
except ImportError:
    from agent import DQNAgent


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    設定隨機種子以確保實驗的可重複性。
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    解析命令列參數。
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the PyTorch model checkpoint (.pt file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results CSV."
    )
    return parser.parse_args()


def evaluate() -> None:
    """
    Execute the DQN evaluation loop across all benchmark target angles.
    在所有基準目標角度上執行 DQN 評估。
    """
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Establish base seed
    base_seed = 666
    set_seed(base_seed)
    
    print("==================================================")
    print("Starting DQN Greedy Evaluation")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Epsilon: 0.0 (Greedy policy)")
    print("==================================================")
    
    # Initialize the Gymnasium environment in headless mode
    env = G1ElbowTargetEnv(render_mode=None)
    
    # Initialize agent and load checkpoint on CPU
    agent = DQNAgent(
        state_dim=4,
        action_dim=3,
        device="cpu"
    )
    
    try:
        agent.load(checkpoint_path)
        print("Checkpoint loaded successfully!\n")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        env.close()
        sys.exit(1)
        
    # Benchmark target angles as specified in the assignment
    benchmark_goals = [-0.8, -0.4, 0.4, 0.8]
    episodes_per_goal = 5
    
    # Result tracking lists
    episode_results = []
    cumulative_rewards = []
    episode_lengths = []
    final_absolute_errors = []
    success_count = 0
    
    print(f"{'Goal (rad)':<12} | {'Episode':<8} | {'Seed':<6} | {'Success':<8} | {'Reward':<10} | {'Steps':<6} | {'Final Error':<12}")
    print("-" * 80)
    
    # Run evaluation episodes
    for goal in benchmark_goals:
        for ep in range(episodes_per_goal):
            episode_seed = base_seed + ep
            set_seed(episode_seed)
            
            # Reset the environment with the specific target goal angle
            state, info = env.reset(seed=episode_seed, options={"goal_angle": goal})
            
            episode_reward = 0.0
            steps = 0
            
            while True:
                # Epsilon = 0.0 for pure greedy action selection
                action = agent.select_action(state, epsilon=0.0)
                
                # Step environment
                next_state, reward, terminated, truncated, info = env.step(action)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
                    
            # Extract final metrics
            is_success = info.get("is_success", False)
            final_err = info.get("absolute_error", abs(info["goal_angle"] - info["elbow_angle"]))
            
            # Record performance metrics
            if is_success:
                success_count += 1
                
            cumulative_rewards.append(episode_reward)
            episode_lengths.append(steps)
            final_absolute_errors.append(final_err)
            
            # Print status to console
            print(f"{goal:<12.1f} | {ep+1:<8d} | {episode_seed:<6d} | {str(is_success):<8} | {episode_reward:<10.4f} | {steps:<6d} | {final_err:<12.6f}")
            
            episode_results.append({
                "Goal": goal,
                "Episode": ep + 1,
                "Seed": episode_seed,
                "Success": is_success,
                "Cumulative_Reward": episode_reward,
                "Episode_Length": steps,
                "Final_Absolute_Error": final_err,
            })
            
    # Close environmental resources
    env.close()
    
    # Compute overall metrics
    total_episodes = len(benchmark_goals) * episodes_per_goal
    success_rate = (success_count / total_episodes) * 100.0
    mean_reward = np.mean(cumulative_rewards)
    mean_length = np.mean(episode_lengths)
    mean_final_error = np.mean(final_absolute_errors)
    
    # Print summary statistics
    print("-" * 80)
    print("Evaluation Summary (DQN Greedy Policy):")
    print(f"Total Successes: {success_count} / {total_episodes}")
    print(f"Success Rate: {success_rate:.2f}% (Required >= 80.00%)")
    print(f"Mean Cumulative Reward: {mean_reward:.4f}")
    print(f"Mean Episode Length: {mean_length:.2f} steps")
    print(f"Mean Final Absolute Error: {mean_final_error:.6f} rad")
    print("-" * 80)
    
    # Save detailed data to CSV file
    csv_path = output_dir / "eval_results.csv"
    print(f"Saving evaluation results to {csv_path}...")
    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write detailed run headers
            writer.writerow([
                "Goal_Angle_rad",
                "Episode_Index",
                "Seed",
                "Success",
                "Cumulative_Reward",
                "Episode_Length",
                "Final_Absolute_Error"
            ])
            # Write data rows
            for res in episode_results:
                writer.writerow([
                    res["Goal"],
                    res["Episode"],
                    res["Seed"],
                    int(res["Success"]),
                    res["Cumulative_Reward"],
                    res["Episode_Length"],
                    res["Final_Absolute_Error"]
                ])
                
            # Write empty line and summary
            writer.writerow([])
            writer.writerow(["Summary_Metric", "Value"])
            writer.writerow(["Successes", f"{success_count}/{total_episodes}"])
            writer.writerow(["Success_Rate_Percent", f"{success_rate:.2f}"])
            writer.writerow(["Mean_Cumulative_Reward", f"{mean_reward:.4f}"])
            writer.writerow(["Mean_Episode_Length", f"{mean_length:.2f}"])
            writer.writerow(["Mean_Final_Absolute_Error_rad", f"{mean_final_error:.6f}"])
            
        print("Results successfully saved!")
    except Exception as e:
        print(f"Failed to save CSV file: {e}")
        
    print("==================================================")


if __name__ == "__main__":
    evaluate()
