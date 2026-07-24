import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    Q-Network for the DQN agent.
    DQN 代理人的 Q 網路。

    MLP Architecture:
        Input (4) -> Linear(64) + ReLU -> Linear(64) + ReLU -> Linear(3)

    The output represents the unconstrained action Q-values (no Softmax in the last layer).
    輸出為無約束的動作 Q 值（最後一層絕不使用 Softmax）。
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 3):
        super(QNetwork, self).__init__()
        # Input layer to the first hidden layer (4 -> 64)
        # 輸入層至第一隱藏層 (4 -> 64)
        self.fc1 = nn.Linear(state_dim, 64)
        
        # First hidden layer to the second hidden layer (64 -> 64)
        # 第一隱藏層至第二隱藏層 (64 -> 64)
        self.fc2 = nn.Linear(64, 64)
        
        # Second hidden layer to the output layer (64 -> 3)
        # 第二隱藏層至輸出層 (64 -> 3)
        self.fc3 = nn.Linear(64, action_dim)
        
        # Activation function ReLU
        # 活化函數 ReLU
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        網路的前向傳播。

        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, state_dim) or (state_dim,)
                              形狀為 (batch_size, state_dim) 或 (state_dim,) 的輸入狀態張量。

        Returns:
            torch.Tensor: Q-values for each action.
                          每個動作對應的 Q 值。
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # No activation function at the output layer
        # 輸出層不上任何活化函數
        q_values = self.fc3(x)
        return q_values
