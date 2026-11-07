"""易读的姿态坐标、骨骼、速度和关键帧损失。"""
import torch.nn.functional as F


def coordinate_loss(prediction, target):
    return F.mse_loss(prediction, target)


def bone_vector_loss(prediction, target):
    pred_bones = prediction[:, :, 1:] - prediction[:, :, :-1]
    target_bones = target[:, :, 1:] - target[:, :, :-1]
    return F.mse_loss(pred_bones, target_bones)


def velocity_loss(prediction, target):
    pred_velocity = prediction[:, 1:] - prediction[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    return F.mse_loss(pred_velocity, target_velocity)


def keyframe_loss(prediction, target, keyframe_mask):
    mask = keyframe_mask.unsqueeze(-1)
    error = (prediction - target).pow(2) * mask
    return error.sum() / mask.sum().clamp_min(1.0) / prediction.shape[2] / 2