"""基础 MLP 以及逐步加入时序、条件和多头结构的模型。"""
import torch
from torch import nn
from src.data import NUM_FRAMES, POSE_DIM


class PoseMLP(nn.Module):
    def __init__(self, condition_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_FRAMES * POSE_DIM),
        )

    def forward(self, condition):
        output = self.network(condition)
        return output.reshape(-1, NUM_FRAMES, POSE_DIM // 2, 2)


class TemporalPoseNet(nn.Module):
    def __init__(self, condition_dim, hidden_dim=64):
        super().__init__()
        self.frame_mlp = nn.Sequential(
            nn.Linear(condition_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, POSE_DIM),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(POSE_DIM, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, POSE_DIM, 3, padding=1),
        )

    def forward(self, condition):
        batch = condition.shape[0]
        time = torch.linspace(0, 1, NUM_FRAMES, device=condition.device)
        time = time.view(1, NUM_FRAMES, 1).expand(batch, -1, -1)
        repeated = condition.unsqueeze(1).expand(-1, NUM_FRAMES, -1)
        coarse = self.frame_mlp(torch.cat([repeated, time], dim=-1))
        residual = self.temporal(coarse.transpose(1, 2)).transpose(1, 2)
        pose = coarse + residual
        return pose.reshape(-1, NUM_FRAMES, POSE_DIM // 2, 2)


class MultiHeadPoseNet(nn.Module):
    def __init__(self, condition_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.ReLU())
        self.body_head = nn.Linear(hidden_dim, NUM_FRAMES * 3 * 2)
        self.hand_head = nn.Linear(hidden_dim, NUM_FRAMES * 2 * 2)

    def forward(self, condition):
        hidden = self.shared(condition)
        body = self.body_head(hidden).reshape(-1, NUM_FRAMES, 3, 2)
        hand = self.hand_head(hidden).reshape(-1, NUM_FRAMES, 2, 2)
        return torch.cat([body, hand], dim=2)


class KeyframePoseNet(nn.Module):
    def __init__(self, condition_dim, keyframe_input_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(condition_dim + keyframe_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_FRAMES * POSE_DIM),
        )

    def forward(self, inputs):
        output = self.network(inputs)
        return output.reshape(-1, NUM_FRAMES, POSE_DIM // 2, 2)


class NoisePredictor(nn.Module):
    def __init__(self, condition_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(NUM_FRAMES * POSE_DIM + condition_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_FRAMES * POSE_DIM),
        )

    def forward(self, noisy_pose, condition, time_fraction):
        inputs = torch.cat([noisy_pose, condition, time_fraction.unsqueeze(1)], dim=1)
        return self.network(inputs)