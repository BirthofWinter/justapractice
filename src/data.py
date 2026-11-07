"""合成二维手臂动作：让练习不依赖外部手语数据集。"""
import math
import numpy as np
import torch
from torch.utils.data import Dataset

GLOSS_NAMES = ("抬手", "挥手", "指向")
NUM_GLOSSES = 3
NUM_FRAMES = 24
NUM_JOINTS = 5
POSE_DIM = NUM_JOINTS * 2


class ToySignDataset(Dataset):
    def __init__(self, size=1024, seed=0):
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        gloss = int(rng.integers(0, NUM_GLOSSES))
        side = float(rng.choice([-1.0, 1.0]))
        scale = float(rng.uniform(0.85, 1.15))
        root_shift = float(rng.uniform(-0.06, 0.06))
        speed = float(rng.uniform(0.8, 1.25))
        pose = make_pose(gloss, side, scale, root_shift, speed)
        extra = np.array(
            [side, (scale - 1.0) / 0.15, root_shift / 0.06],
            dtype=np.float32,
        )
        return {
            "gloss": torch.tensor(gloss, dtype=torch.long),
            "extra": torch.from_numpy(extra),
            "pose": torch.from_numpy(pose),
        }


def make_pose(gloss, side=1.0, scale=1.0, root_shift=0.0, speed=1.0):
    frames = np.linspace(0.0, 1.0, NUM_FRAMES, dtype=np.float32)
    phases = np.power(frames, speed)
    lengths = np.array([0.20, 0.18, 0.12, 0.10], dtype=np.float32) * scale
    poses = np.zeros((NUM_FRAMES, NUM_JOINTS, 2), dtype=np.float32)

    for frame_index, phase in enumerate(phases):
        if gloss == 0:
            angles = [0.65 - 1.8 * phase, 0.10, 0.08, 0.05]
        elif gloss == 1:
            angles = [
                -0.35,
                0.25,
                0.25 + 0.65 * math.sin(2.0 * math.pi * 2.0 * phase),
                0.10,
            ]
        else:
            angles = [-0.15, -0.10, -0.05, -0.10]

        x, y = 0.50 + root_shift, 0.78
        poses[frame_index, 0] = (x, y)
        for joint_index, (length, angle) in enumerate(zip(lengths, angles), start=1):
            x += side * float(length) * math.cos(angle)
            y += float(length) * math.sin(angle)
            poses[frame_index, joint_index] = (x, y)
    return poses


def make_gloss_one_hot(gloss):
    return torch.nn.functional.one_hot(gloss, num_classes=NUM_GLOSSES).float()


def make_keyframe_inputs(pose):
    batch, frames, joints, xy = pose.shape
    mask = torch.zeros(batch, frames, 1, device=pose.device)
    indices = (0, frames // 2, frames - 1)
    mask[:, list(indices), :] = 1.0
    sparse_pose = pose * mask.unsqueeze(-1)
    return sparse_pose.reshape(batch, -1), mask.reshape(batch, -1), mask