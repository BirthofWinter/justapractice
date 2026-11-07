"""动作轨迹和训练曲线的绘图工具。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BONES = ((0, 1), (1, 2), (2, 3), (3, 4))


def draw_pose_sequence(axis, pose, title):
    pose = np.asarray(pose)
    frames = np.linspace(0, len(pose) - 1, 6, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(frames)))
    for frame, color in zip(frames, colors):
        points = pose[frame]
        for start, end in BONES:
            axis.plot(
                [points[start, 0], points[end, 0]],
                [points[start, 1], points[end, 1]],
                color=color, alpha=0.75, linewidth=2,
            )
        axis.scatter(points[:, 0], points[:, 1], color=[color], s=18)
    axis.set_title(title)
    axis.set_xlim(0, 1)
    axis.set_ylim(1, 0)
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.2)


def save_pose_comparison(target, prediction, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    draw_pose_sequence(axes[0], target, "目标动作")
    draw_pose_sequence(axes[1], prediction, "模型预测")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def save_loss_curve(train_losses, val_losses, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(train_losses, label="train")
    axis.plot(val_losses, label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)