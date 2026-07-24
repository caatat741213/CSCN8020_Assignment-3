import sys
import os
from pathlib import Path

# Add 'src' directory to Python path for importing modules
# 將 'src' 目錄加入 Python 路徑，以匯入相關模組
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

import numpy as np
import torch
from g1_rl.g1_elbow_env import G1ElbowTargetEnv
from dqn.agent import DQNAgent

def main() -> None:
    print("==================================================")
    print("Starting DQN Module Smoke Test...")
    #print("開始 DQN 模組冒煙測試...")
    print("==================================================")

    # 1. Environment initialization / 初始化環境
    print("\n[Step 1] Initializing G1ElbowTargetEnv...")
    try:
        env = G1ElbowTargetEnv(render_mode=None)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        print(f"   Environment loaded successfully.")
        print(f"   State space dimension: {state_dim}")
        print(f"   Action space dimension: {action_dim}")
    except Exception as e:
        print(f"   Failed to load G1ElbowTargetEnv: {e}")
        sys.exit(1)

    # 2. Agent initialization / 初始化代理人
    print("\n[Step 2] Initializing DQNAgent with test hyperparameters...")
    try:
        # Use small batch size and warmup for quick smoke test validation
        # 使用較小的批次大小與預熱步數，以便快速進行冒煙測試驗證
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=0.95,
            lr=0.001,
            batch_size=8,
            target_update=10,
            warmup=10,
            buffer_capacity=1000,
            device="cpu"
        )
        print(f"   DQNAgent initialized on device: {agent.device}")
    except Exception as e:
        print(f"   Failed to initialize DQNAgent: {e}")
        env.close()
        sys.exit(1)

    # 3. Running 5 episodes / 執行 5 個回合
    print("\n[Step 3] Running 5 episodes to collect experiences...")
    num_episodes = 5
    total_steps = 0
    try:
        for ep in range(num_episodes):
            observation, info = env.reset(seed=42 + ep)
            ep_reward = 0.0
            ep_steps = 0
            
            while True:
                # Epsilon-greedy action selection
                # 使用 epsilon-greedy 選擇動作
                action = agent.select_action(observation, epsilon=0.2)
                
                # Step the environment
                # 執行環境步進
                next_observation, reward, terminated, truncated, info = env.step(action)
                
                # Save to replay buffer (only terminated is stored as done)
                # 存入重放緩衝區（僅將 terminated 存為 done）
                agent.replay_buffer.push(observation, action, reward, next_observation, terminated)
                
                observation = next_observation
                ep_reward += reward
                ep_steps += 1
                total_steps += 1
                
                if terminated or truncated:
                    break
                    
            print(f"   Episode {ep + 1}: Steps = {ep_steps}, Reward = {ep_reward:.4f}, Terminated = {terminated}, Truncated = {truncated}")
            
        print(f"   Total steps collected: {total_steps}")
        print(f"   Replay buffer size: {len(agent.replay_buffer)}")
        assert len(agent.replay_buffer) == total_steps, "Buffer size does not match collected steps!"
        print("   Replay buffer push test passed.")
    except Exception as e:
        print(f"   Error during episode execution or buffer pushing: {e}")
        env.close()
        sys.exit(1)

    # 4. Batch sampling and optimization test / 批次採樣與最佳化測試
    print("\n[Step 4] Testing batch sampling, loss computation, and parameter update...")
    try:
        # Check sampling
        # 檢查採樣
        batch_size = agent.batch_size
        states, actions, rewards, next_states, terminateds = agent.replay_buffer.sample(batch_size)
        print(f"   Sampled batch shapes:")
        print(f"     states:     {states.shape}")
        print(f"     actions:    {actions.shape}")
        print(f"     rewards:    {rewards.shape}")
        print(f"     next_states:{next_states.shape}")
        print(f"     terminateds:{terminateds.shape}")
        
        assert states.shape == (batch_size, state_dim), "States shape mismatch!"
        assert actions.shape == (batch_size, 1), "Actions shape mismatch!"
        assert rewards.shape == (batch_size, 1), "Rewards shape mismatch!"
        
        # Check optimization (1 step of gradient descent)
        # 檢查最佳化（執行 1 步梯度下降）
        loss = agent.optimize_model()
        print(f"   Optimization complete. Computed Loss = {loss:.6f}")
        assert loss is not None, "Optimization returned None!"
        assert loss >= 0, f"Invalid loss computed: {loss}"
        print("   Batch sampling, loss calculation, and gradient update test passed.")
    except Exception as e:
        print(f"   Error during optimization test: {e}")
        env.close()
        sys.exit(1)

    # 5. Checkpoint saving and loading / 檢查點儲存與載入
    print("\n[Step 5] Testing model save and load checkpoint...")
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)
    test_checkpoint = checkpoint_dir / "test_smoke_checkpoint.pt"
    
    try:
        # Save model
        # 儲存模型
        agent.save(test_checkpoint)
        print(f"   Checkpoint saved to: {test_checkpoint}")
        assert test_checkpoint.exists(), "Checkpoint file was not created!"
        
        # Load model into a clean agent
        # 將模型載入至新的代理人中
        new_agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")
        new_agent.load(test_checkpoint)
        print("   Checkpoint loaded into a new agent successfully.")
        
        # Clean up temporary test checkpoint file
        # 清除暫存的測試檢查點檔案
        if test_checkpoint.exists():
            test_checkpoint.unlink()
            print("   Temporary checkpoint file deleted.")
        print("   Save/Load checkpoint test passed.")
    except Exception as e:
        print(f"   Error during checkpoint save/load test: {e}")
        if test_checkpoint.exists():
            test_checkpoint.unlink()
        env.close()
        sys.exit(1)

    # Close environment
    # 關閉模擬環境
    env.close()
    
    print("\n==================================================")
    print("ALL DQN MODULE SMOKE TESTS PASSED SUCCESSFULLY!")
    #print("所有 DQN 模組冒煙測試皆成功通過！")
    print("==================================================")

if __name__ == "__main__":
    main()
