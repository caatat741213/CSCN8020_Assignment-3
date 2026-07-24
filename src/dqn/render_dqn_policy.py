"""
This script loads the selected pre-trained DQN model and renders the policy
It runs 1 episode for each of the 4 benchmark angles with epsilon = 0.0.

載入預訓練模型並以無探索 (epsilon = 0.0) 在 Gymnasium GUI 中輪流演示 4 個基準目標角度。
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Add 'src' directory to Python path to import g1_rl and dqn modules
# 將 'src' 目錄加入 Python 路徑
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from g1_rl import G1ElbowTargetEnv
from dqn.agent import DQNAgent


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    設定隨機種子以確保可重複性。
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main() -> None:
    # 1. Path Setup for Pre-trained Model / 設定預訓練模型路徑
    # Try local root folder first, then relative to script path
    # 優先嘗試專案根目錄的路徑，若無則使用相對於腳本的路徑
    checkpoint_name = "selected_dqn.pt"
    possible_paths = [
        Path("models") / checkpoint_name,
        current_dir.parent.parent / "models" / checkpoint_name,
    ]
    
    checkpoint_path = None
    for path in possible_paths:
        if path.is_file():
            checkpoint_path = path
            break
            
    if checkpoint_path is None:
        print(f"Error: Pre-trained model '{checkpoint_name}' not found!")
        print("Please ensure it exists under the 'models/' directory.")
        sys.exit(1)
        
    print("==================================================")
    print("DQN Policy GUI Render & Recording Script")
    print(f"Loading pre-trained model: {checkpoint_path}")
    print("Epsilon: 0.0 (Pure greedy, no exploration)")
    print("==================================================")
    
    # Set seed for reproducibility / 設定種子以確保再現性
    set_seed(666)
    
    # 2. Initialize Gymnasium Environment with GUI Render Mode / 初始化環境
    # render_mode="human" launches the interactive MuJoCo passive viewer
    env = G1ElbowTargetEnv(render_mode="human")
    
    # 3. Initialize Agent & Load Checkpoint / 初始化代理人並載入權重
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device="cpu"  # Force CPU execution as required
    )
    
    try:
        agent.load(checkpoint_path)
        print("DQN model weights loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        env.close()
        sys.exit(1)
        
    # 4. Benchmark angles to test sequentially / 輪流測試 the benchmark angles
    benchmark_goals = [-0.8, -0.4, 0.4, 0.8]
    total_episodes = len(benchmark_goals)
    
    # 5. Render Evaluation Loop / 視覺化評估迴圈
    for ep_idx, target_angle in enumerate(benchmark_goals, 1):
        # Reset environment with specific target angle first.
        # This will launch the MuJoCo viewer GUI and render the initial state.
        # 先進行環境重置，這會啟動 MuJoCo 檢視器 GUI 並渲染初始狀態。
        state, info = env.reset(seed=666 + ep_idx, options={"goal_angle": target_angle})

        if ep_idx == 1:
            # Prompt user to adjust the camera and start recording after GUI window has opened
            # 在 GUI 視窗開啟後，提示使用者調整畫面與準備錄影
            print("\n" + "=" * 60)
            print("MuJoCo GUI window is now open.")
            print("1. Please adjust the camera view in the MuJoCo simulator window.")
            print("2. Start your screen/video recording software when ready.")
            print("3. Press [Enter] in this terminal to start Episode 1.")
            print("=" * 60)
            input("Press [Enter] to start Episode 1...")
            print("\nStarting simulation loop...\n")
        else:
            # Pause for subsequent episodes to allow camera adjustments for the new target
            # 對於後續的回合，暫停以允許針對新目標進行相機視角調整
            print("\n" + "=" * 80)
            print(f"Preparing Episode {ep_idx}/{total_episodes} (Target: {target_angle} rad)")
            print("You can adjust the camera now if needed.")
            input(f"Press [Enter] to run Episode {ep_idx}...")
            print("\nStarting simulation...\n")

        print("-" * 80)
        print(f"{'Ep #':<6} | {'Target (rad)':<12} | {'Current (rad)':<12} | {'Step':<6} | {'Reward':<10} | {'Success Status'}")
        print("-" * 80)
        
        step_count = 0
        episode_reward = 0.0
        
        while True:
            # Force epsilon = 0.0 for pure greedy action selection
            # 強制設定 epsilon = 0.0 以執行純貪婪決策
            action = agent.select_action(state, epsilon=0.0)
            
            # Step the environment / 環境步進
            next_state, reward, terminated, truncated, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Retrieve real-time joint positions and success variables
            # 取得即時的手肘角度、目標角度與成功狀態
            elbow_angle = info["elbow_angle"]
            goal_angle = info["goal_angle"]
            streak = info["success_streak"]
            is_success = info.get("is_success", terminated)
            
            # Construct success status string
            # 建立易讀的成功狀態字串
            success_status_str = f"{is_success} (Streak: {streak}/8)"
            if is_success:
                success_status_str = f"SUCCESS (Streak: {streak}/8)"
                
            # Real-time console print for each step
            # 終端機即時印出每一步的狀態
            print(
                f"{ep_idx:<6d} | "
                f"{goal_angle:<12.4f} | "
                f"{elbow_angle:<12.4f} | "
                f"{step_count:<6d} | "
                f"{reward:<10.4f} | "
                f"{success_status_str}"
            )
            
            if terminated or truncated:
                break
                
        print("-" * 80)
        print(f"Episode {ep_idx} Finished!")
        print(f"Target Angle: {target_angle:+.2f} rad")
        print(f"Total Steps: {step_count}")
        print(f"Cumulative Reward: {episode_reward:.4f}")
        print(f"Final Success State: {is_success}")
        print("=" * 80 + "\n")
        
        # Pause briefly between episodes to let the visual settle
        # 回合間短暫停頓以方便視覺觀看
        time.sleep(1.0)
        
    print("All benchmark target angles rendered successfully!")
    
    # Prompt the user on how to close the viewer
    # 提醒用戶如何關閉檢視器視窗
    if env.viewer is not None:
        print("Close the MuJoCo viewer window to finish.")
        while env.viewer is not None and env.viewer.is_running():
            time.sleep(0.1)
            
    env.close()


if __name__ == "__main__":
    main()
