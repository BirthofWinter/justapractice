"""分阶段实验入口：先跑 baseline，再一次尝试一个新模块。"""
import argparse
import csv
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import (
    GLOSS_NAMES, NUM_GLOSSES, NUM_FRAMES, POSE_DIM,
    ToySignDataset, make_gloss_one_hot, make_keyframe_inputs,
)
from src.diffusion import add_noise, make_noise_schedule, sample_pose
from src.losses import (
    bone_vector_loss, coordinate_loss, keyframe_loss, velocity_loss,
)
from src.models import (
    KeyframePoseNet, MultiHeadPoseNet, NoisePredictor,
    PoseMLP, TemporalPoseNet,
)
from src.visualize import save_loss_curve, save_pose_comparison

EXPERIMENTS = (
    "baseline", "bone", "temporal", "diffusion",
    "multi_condition", "multi_head", "keyframe",
)


def parse_args():
    parser = argparse.ArgumentParser(description="合成二维手语动作 PyTorch 练习")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--bone_weight", type=float, default=0.25)
    parser.add_argument("--smooth_weight", type=float, default=0.10)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", default="outputs")
    return parser.parse_args()


def build_condition(batch, experiment):
    gloss = make_gloss_one_hot(batch["gloss"])
    if experiment == "multi_condition":
        # 五档量化是 FSQ 思路的课堂演示，不是完整 FSQ 自编码器。
        continuous = batch["extra"].clamp(-1.0, 1.0)
        quantized = torch.round((continuous + 1.0) * 2.0) / 2.0 - 1.0
        return torch.cat([gloss, quantized], dim=1)
    return gloss


def build_model(experiment):
    if experiment in ("baseline", "bone"):
        return PoseMLP(NUM_GLOSSES)
    if experiment == "temporal":
        return TemporalPoseNet(NUM_GLOSSES)
    if experiment == "diffusion":
        return NoisePredictor(NUM_GLOSSES)
    if experiment == "multi_condition":
        return PoseMLP(NUM_GLOSSES + 3)
    if experiment == "multi_head":
        return MultiHeadPoseNet(NUM_GLOSSES)
    if experiment == "keyframe":
        key_dim = NUM_FRAMES * POSE_DIM + NUM_FRAMES
        return KeyframePoseNet(NUM_GLOSSES, key_dim)
    raise ValueError(f"未知实验：{experiment}")


def prepare_input(batch, experiment):
    condition = build_condition(batch, experiment)
    if experiment == "keyframe":
        sparse_pose, mask_flat, mask = make_keyframe_inputs(batch["pose"])
        return torch.cat([condition, sparse_pose, mask_flat], dim=1), mask
    return condition, None


def pose_objective(prediction, target, experiment, args, mask):
    loss = coordinate_loss(prediction, target)
    if experiment == "bone":
        loss = loss + args.bone_weight * bone_vector_loss(prediction, target)
    if experiment == "temporal":
        loss = loss + args.smooth_weight * velocity_loss(prediction, target)
    if experiment == "keyframe":
        loss = loss + keyframe_loss(prediction, target, mask)
    return loss


def run_epoch(model, loader, optimizer, experiment, args, device, schedule):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        target = batch["pose"]
        if experiment == "diffusion":
            clean = target.reshape(target.shape[0], -1)
            noise = torch.randn_like(clean)
            time_index = torch.randint(
                0, args.diffusion_steps, (target.shape[0],), device=device
            )
            noisy = add_noise(clean, noise, time_index, schedule[2])
            condition = make_gloss_one_hot(batch["gloss"])
            time = time_index.float() / max(args.diffusion_steps - 1, 1)
            prediction = model(noisy, condition, time)
            loss = F.mse_loss(prediction, noise)
        else:
            model_input, mask = prepare_input(batch, experiment)
            prediction = model(model_input)
            loss = pose_objective(prediction, target, experiment, args, mask)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        count = target.shape[0]
        total_loss += loss.detach().item() * count
        total_samples += count

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def make_preview(model, dataset, experiment, args, device):
    sample = dataset[0]
    batch = {
        name: value.unsqueeze(0).to(device)
        for name, value in sample.items()
    }
    target = batch["pose"]

    if experiment == "diffusion":
        condition = make_gloss_one_hot(batch["gloss"])
        flat = sample_pose(
            model, condition, args.diffusion_steps,
            NUM_FRAMES * POSE_DIM, device,
        )
        prediction = flat.reshape(1, NUM_FRAMES, POSE_DIM // 2, 2)
    else:
        model_input, _ = prepare_input(batch, experiment)
        prediction = model(model_input)

    return target[0].cpu().numpy(), prediction[0].cpu().numpy(), int(sample["gloss"])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = ToySignDataset(args.samples, seed=args.seed)
    val_data = ToySignDataset(max(args.samples // 4, 128), seed=args.seed + 100_000)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.experiment).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    schedule = make_noise_schedule(args.diffusion_steps, device)

    train_losses, val_losses = [], []
    print(f"实验：{args.experiment} | 设备：{device}")
    print(f"动作类别：{', '.join(GLOSS_NAMES)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, args.experiment, args, device, schedule
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model, val_loader, None, args.experiment, args, device, schedule
            )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.experiment}_loss.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, values in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([epoch, *values])

    target, prediction, gloss_id = make_preview(
        model, val_data, args.experiment, args, device
    )
    image_path = output_dir / f"{args.experiment}_preview.png"
    save_pose_comparison(
        target, prediction, image_path,
        f"{args.experiment}: {GLOSS_NAMES[gloss_id]}",
    )
    curve_path = output_dir / f"{args.experiment}_curve.png"
    save_loss_curve(train_losses, val_losses, curve_path, args.experiment)

    print(f"损失记录：{csv_path}")
    print(f"动作对比图：{image_path}")
    print(f"训练曲线：{curve_path}")


if __name__ == "__main__":
    main()