import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from dqn.q_network import QNetwork
    from dqn.replay_buffer import ReplayBuffer
except ImportError:
    from q_network import QNetwork
    from replay_buffer import ReplayBuffer

class DQNAgent:
    """
    DQN Agent managing training, action selection, and optimization.
    DQN 代理人，管理訓練、動作選擇與最佳化。
    """
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 3,
        gamma: float = 0.95,
        lr: float = 0.001,
        batch_size: int = 64,
        target_update: int = 250,
        warmup: int = 500,
        buffer_capacity: int = 50000,
        device: str | None = None
    ):
        """
        Initialize the DQN Agent.
        初始化 DQN 代理人。
        """
        # Determine device / 決定執行裝置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set hyperparameters / 設定超參數
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.warmup = warmup

        # Initialize online and target Q-networks / 初始化線上與目標 Q 網路
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        
        # Hard sync target network at start / 起始時硬同步目標網路
        self.update_target_network()
        self.target_net.eval()  # Target net is only used for evaluation / 目標網路僅在求取目標值時使用

        # Initialize replay buffer / 初始化重放緩衝區
        self.replay_buffer = ReplayBuffer(buffer_capacity, self.device)

        # Optimizer / 最佳化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Loss function (Huber Loss is default for stability) / 損失函數（預設使用較穩定的 Huber 損失）
        self.loss_fn = nn.SmoothL1Loss()

        # Step counter for optimization steps / 最佳化步驟計數器
        self.opt_steps = 0
        
        # Step counter for total environment steps / 總環境步數計數器
        self.env_steps = 0

    def update_target_network(self) -> None:
        """
        Synchronize target network with online network.
        同步目標網路與線上網路的參數。
        """
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action using epsilon-greedy exploration.
        使用 epsilon-greedy 探索策略選擇動作。

        Args:
            state (np.ndarray): The current observation state.
                                目前的狀態觀察值。
            epsilon (float): Exploration rate. Default is 0.0 (greedy action).
                             探索率。預設為 0.0（採用貪婪動作）。

        Returns:
            int: The chosen action (0, 1, or 2).
                 選取的動作。
        """
        self.env_steps += 1
        
        # Epsilon-greedy check
        # Epsilon-greedy 檢查
        if np.random.rand() < epsilon:
            # Explore: random action
            # 探索：隨機選擇動作
            return int(np.random.randint(3))
        else:
            # Exploit: greedy action
            # 利用：選擇 Q 值最大的動作
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # Add batch dimension / 新增批次維度
            
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                return int(q_values.argmax(dim=1).item())

    def optimize_model(self) -> float | None:
        """
        Perform one step of gradient descent optimization.
        執行一步梯度下降最佳化。

        Returns:
            float | None: Loss value if optimized, None otherwise.
                          若執行了最佳化則回傳損失值，否則回傳 None。
        """
        # Do not optimize if buffer does not contain enough samples
        # 如果重放緩衝區中的樣本不足，則不進行最佳化
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample transitions from replay buffer
        # 從重放緩衝區中採樣
        states, actions, rewards, next_states, terminateds = self.replay_buffer.sample(self.batch_size)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        # 計算當前的 Q 值 Q(s_t, a)
        current_q = self.q_net(states).gather(1, actions)

        # Compute max_a Q_target(s_{t+1}, a) for next states.
        # 計算下一狀態的最大預期 Q 值
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            
            # Bootstrapping rule: only terminated == True does not bootstrap (terminateds is 1.0)
            # Truncated is handled as not terminated, allowing bootstrapping (terminateds is 0.0)
            # 自舉規則：只有當 terminated == True 時不進行自舉，其餘情況仍照常自舉
            target_q = rewards + self.gamma * max_next_q * (1.0 - terminateds)

        # Compute Huber loss
        # 計算 Huber 損失
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        # 進行參數更新
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping with max_norm = 1.0
        # 梯度裁剪，最大範數為 1.0
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Increment optimization steps and check for target net update
        # 增加最佳化次數，並檢查是否需要同步目標網路
        self.opt_steps += 1
        if self.opt_steps % self.target_update == 0:
            self.update_target_network()

        return float(loss.item())

    def save(self, filepath: str | Path) -> None:
        """
        Save the model checkpoint.
        儲存模型檢查點。
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_net_state_dict": self.q_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "opt_steps": self.opt_steps,
            "env_steps": self.env_steps,
        }, str(filepath))

    def load(self, filepath: str | Path) -> None:
        """
        Load the model checkpoint.
        載入模型檢查點。
        """
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"No checkpoint file found at {filepath}")
        
        checkpoint = torch.load(str(filepath), map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.opt_steps = checkpoint.get("opt_steps", 0)
        self.env_steps = checkpoint.get("env_steps", 0)
        # Ensure target network is synced on load
        # 確保載入後目標網路是同步的
        self.update_target_network()
