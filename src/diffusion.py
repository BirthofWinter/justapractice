"""最小化的 DDPM 教学实现。"""
import torch


def make_noise_schedule(steps, device):
    beta = torch.linspace(1e-4, 0.02, steps, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def add_noise(clean_pose, noise, time_index, alpha_bar):
    selected = alpha_bar[time_index].unsqueeze(1)
    return selected.sqrt() * clean_pose + (1.0 - selected).sqrt() * noise


@torch.no_grad()
def sample_pose(model, condition, steps, pose_dim, device):
    beta, alpha, alpha_bar = make_noise_schedule(steps, device)
    pose = torch.randn(condition.shape[0], pose_dim, device=device)
    model.eval()

    for step in reversed(range(steps)):
        time = torch.full(
            (condition.shape[0],), step / max(steps - 1, 1), device=device
        )
        predicted_noise = model(pose, condition, time)
        beta_t = beta[step]
        alpha_t = alpha[step]
        alpha_bar_t = alpha_bar[step]
        pose = pose - beta_t / torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
        pose = pose / torch.sqrt(alpha_t)
        if step > 0:
            pose = pose + torch.sqrt(beta_t) * torch.randn_like(pose)
    return pose