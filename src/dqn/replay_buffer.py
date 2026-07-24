import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """
    Replay Buffer for storing and sampling experience transitions.
    用於儲存與採樣經驗轉移資料的重放緩衝區。
    """
    def __init__(self, capacity: int = 50000, device: torch.device | str = "cpu"):
        """
        Initialize the Replay Buffer.
        初始化重放緩衝區。

        Args:
            capacity (int): Maximum number of transitions to store. Default is 50,000.
                            最大可儲存之狀態轉換筆數。預設為 50,000。
            device (torch.device | str): Device to place the sampled PyTorch tensors on.
                                         採樣後的 PyTorch 張量放置的裝置。
        """
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device(device)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a new transition to the buffer.
        新增一筆新的狀態轉換至緩衝區。

        Args:
            state (np.ndarray): Current state observation.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state observation.
            done (bool): Terminal signal (terminated).
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a mini-batch of transitions randomly.
        隨機採樣一小批次的狀態轉換資料。

        Args:
            batch_size (int): Size of the batch to sample.
                              採樣批次的大小。

        Returns:
            tuple[torch.Tensor, ...]: Tensors of (states, actions, rewards, next_states, dones)
                                      包含 (states, actions, rewards, next_states, dones) 的張量元組。
        """
        # Randomly sample transitions from deque
        # 從 deque 中隨機採樣
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip into separate tuples
        # 拆解為個別的元組
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch Tensors on the designated device
        # 轉換為指定裝置上的 PyTorch 張量
        states_tensor = torch.tensor(
            np.array(states), dtype=torch.float32, device=self.device
        )
        actions_tensor = torch.tensor(
            actions, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states_tensor = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=self.device
        )
        dones_tensor = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )

    def __len__(self) -> int:
        """
        Return the current size of the buffer.
        回傳目前緩衝區的長度。
        """
        return len(self.buffer)
