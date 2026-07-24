"""
DQN Training Script for Unitree G1 Left Elbow Control.
This script supports command line arguments to choose configurations and episodes.
All console prints are in English as required.

CSCN8020 DQN 訓練與消融實驗腳本。
支援選擇 Config A, B, C, D, E 及 Episodes 回合數。
"""

from __future__ import annotations
import os
import sys
import time
import csv
import random
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
    Set all random seeds to guarantee reproducibility.
    設定隨機種子以確保實驗的可重複性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Force PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    解析命令列參數。
    """
    parser = argparse.ArgumentParser(description="Train a DQN agent for Unitree G1 elbow control.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["a", "b", "c_linear", "d_fast_target", "e_small_buffer"],
        help="Configuration configuration to run (a, b, c_linear, d_fast_target, e_small_buffer)."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=700,
        help="Total number of training episodes. Default is 700."
    )
    return parser.parse_args()


def train() -> None:
    """
    Execute the DQN training loop, record metrics to CSV, and save checkpoints.
    執行 DQN 訓練迴圈，記錄指標至 CSV 並儲存檢查點。
    """
    args = parse_args()
    config_name = f"config_{args.config}"
    
    # Establish base seed for reproducibility
    base_seed = 666
    set_seed(base_seed)
    
    # Set configuration hyperparameters
    epsilon_decay = 0.995
    target_update = 250
    buffer_capacity = 50000
    is_linear_decay = False
    
    if args.config == "a":
        epsilon_decay = 0.995
    elif args.config == "b":
        epsilon_decay = 0.985
    elif args.config == "c_linear":
        is_linear_decay = True
    elif args.config == "d_fast_target":
        target_update = 50
    elif args.config == "e_small_buffer":
        buffer_capacity = 1000
        
    print("==================================================")
    print(f"Starting DQN Training: {config_name.upper()}")
    print(f"Total Episodes: {args.episodes}")
    print(f"Random Seed: {base_seed}")
    print(f"Hyperparameters:")
    print(f" - Epsilon Decay: {'Linear over 500 ep' if is_linear_decay else epsilon_decay}")
    print(f" - Target Update Frequency: {target_update} steps")
    print(f" - Buffer Capacity: {buffer_capacity} transitions")
    print("==================================================")
    
    # Initialize the Gymnasium environment in headless mode
    # Pass goal_range=(-0.8, 0.8) as required by the multi-goal training scope
    env = G1ElbowTargetEnv(
        render_mode=None,
        goal_range=(-0.8, 0.8)
    )
    
    # Initialize the DQN agent
    # Device is set to CPU as required for compatibility and headless execution
    agent = DQNAgent(
        state_dim=4,
        action_dim=3,
        gamma=0.95,
        lr=0.001,
        batch_size=64,
        target_update=target_update,
        warmup=500,
        buffer_capacity=buffer_capacity,
        device="cpu"
    )
    
    # Setup exploration parameters
    epsilon = 1.0
    epsilon_min = 0.05
    
    # Define directories and filepaths
    results_dir = Path("results") / config_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log_csv_path = results_dir / "training_log.csv"
    checkpoint_path = Path("models") / f"{config_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup CSV writing
    csv_headers = [
        "Episode",
        "Reward",
        "Success",
        "Steps",
        "Final_Absolute_Error",
        "Epsilon",
        "Loss",
        "Wall_Clock_Time"
    ]
    
    start_time = time.time()
    success_history = []
    reward_history = []
    
    # Open CSV file and write headers
    with open(log_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        
        print(f"{'Episode':<8} | {'Reward':<10} | {'Success':<8} | {'Steps':<6} | {'Final Error':<12} | {'Epsilon':<8} | {'Avg Loss':<10} | {'Time (s)':<8}")
        print("-" * 90)
        
        for ep in range(1, args.episodes + 1):
            # Calculate linear decay epsilon if config C
            if is_linear_decay:
                # Epsilon linearly decays from 1.0 to 0.05 over 500 episodes
                epsilon = max(epsilon_min, 1.0 - (ep - 1) * (1.0 - epsilon_min) / 500)
                
            # Seed the episode reset deterministically based on episode number
            # to guarantee reproducibility across different runs
            episode_seed = base_seed + ep
            state, info = env.reset(seed=episode_seed)
            
            episode_reward = 0.0
            episode_steps = 0
            episode_losses = []
            
            while True:
                # Epsilon-greedy action selection
                action = agent.select_action(state, epsilon)
                
                # Perform step in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Store transition in replay buffer
                # Pass terminated (success) as 'done' so we bootstrap on time-limit truncation,
                # but NOT on true terminal state success
                agent.replay_buffer.push(state, action, reward, next_state, terminated)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Run optimization step if warmup is complete
                if len(agent.replay_buffer) >= agent.warmup:
                    loss_val = agent.optimize_model()
                    if loss_val is not None:
                        episode_losses.append(loss_val)
                
                # End episode on termination or truncation
                if terminated or truncated:
                    break
            
            # Epsilon used in this episode
            old_epsilon = epsilon
            
            # Decay exploration rate for exponential configs
            if not is_linear_decay:
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # Extract final outcome metrics
            is_success = info.get("is_success", False)
            final_err = info.get("absolute_error", abs(info["goal_angle"] - info["elbow_angle"]))
            avg_loss = np.mean(episode_losses) if len(episode_losses) > 0 else 0.0
            wall_clock_time = time.time() - start_time
            
            success_history.append(int(is_success))
            reward_history.append(episode_reward)
            
            # Write to CSV log
            writer.writerow([
                ep,
                episode_reward,
                int(is_success),
                episode_steps,
                final_err,
                old_epsilon,
                avg_loss,
                wall_clock_time
            ])
            f.flush()  # Ensure data is written immediately
            
            # Console output
            if ep % 20 == 0 or ep == 1:
                rolling_success = np.mean(success_history[-50:]) * 100.0 if len(success_history) >= 50 else np.mean(success_history) * 100.0
                print(f"{ep:<8d} | {episode_reward:<10.4f} | {str(is_success):<8} | {episode_steps:<6d} | {final_err:<12.6f} | {old_epsilon:<8.4f} | {avg_loss:<10.6f} | {wall_clock_time:<8.1f} (Rolling Success: {rolling_success:.1f}%)")
                
    # Close environmental resources
    env.close()
    
    # Save the final model checkpoint
    print("-" * 90)
    print(f"Saving final trained checkpoint to {checkpoint_path}...")
    agent.save(checkpoint_path)
    
    # Summary of final training status
    total_time = time.time() - start_time
    final_rolling_success = np.mean(success_history[-50:]) * 100.0
    mean_reward_final_20 = np.mean(reward_history[-20:])
    print(f"Training completed in {total_time:.1f} seconds.")
    print(f"Final rolling success rate (last 50 episodes): {final_rolling_success:.2f}%")
    print(f"Mean cumulative reward (last 20 episodes): {mean_reward_final_20:.4f}")
    print("==================================================")


if __name__ == "__main__":
    train()
