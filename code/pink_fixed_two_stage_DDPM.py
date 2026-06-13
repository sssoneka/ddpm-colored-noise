import csv
import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_fidelity
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

noise_name = "pink_fixed"

pretrain_dataset_name = "imagenet1k"
finetune_dataset_name = "fractaldb1k"

pretrain_epochs = 100
finetune_epochs = 100

project_dir = Path.home() / "ddpm_coursework"
data_root = project_dir / "data"
results_root = project_dir / "results_pink_fixed"

imagenet_root = data_root / "imagenet1k"
imagenet_train_dir = imagenet_root / "train"
imagenet_val_dir = imagenet_root / "val"

fractaldb1k_root = data_root / "fractaldb_1k"
fractaldb60_root = data_root / "fractaldb_60"

stage_name = None
dataset_name = None
run_name = None
run_dir = None
checkpoint_dir = None
logs_dir = None
samples_dir = None
metrics_dir = None
preview_dir = None
train_only = None
enable_preview_samples = None
enable_generation_eval = None
enable_fid = None
enable_kid = None
enable_precision_recall = None
epochs = None

image_size = 64
in_channels = 3
base_channels = 64
time_embed_dim = 256
train_timesteps = 1000
beta_start = 1e-4
beta_end = 2e-2

pink_alpha = 1.0
pink_kappa0 = 1.0 / image_size
pink_normalize = False

batch_size = 64
learning_rate = 2e-4
weight_decay = 0.0
num_workers = 8
grad_clip_norm = 1.0
seed = 42

use_mixed_precision = True
mixed_precision_dtype = "fp16"
allow_tf32 = True

preview_every_epochs = 5
preview_num_images = 16
preview_sample_steps = 100

gen_eval_steps = [100, 200, 300, 400, 500]
num_eval_gen_images = 5000
num_eval_real_images = 5000
gen_batch_size = 64
torch_fidelity_batch_size = 64
kid_subsets = 50
kid_subset_size = 1000

val_ratio = 0.1
max_train_samples = None
max_val_samples = None
max_real_eval_images = None
max_gen_eval_images = None

save_last_every_epoch = True
save_best_by_val_loss = True
auto_resume_last = True

load_imagenet_pretrain_for_fractal = True

pin_memory = torch.cuda.is_available()
persistent_workers = num_workers > 0
device = "cuda" if torch.cuda.is_available() else "cpu"


def mixed_precision_enabled():
    return bool(use_mixed_precision and device == "cuda")


def get_amp_dtype():
    if mixed_precision_dtype.lower() == "fp16":
        return torch.float16
    return torch.bfloat16


def amp_autocast():
    if not mixed_precision_enabled():
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=get_amp_dtype())


def make_grad_scaler():
    enabled = mixed_precision_enabled() and mixed_precision_dtype.lower() == "fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_seconds(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
    if minutes > 0:
        return f"{minutes:02d}m {seconds:02d}s"
    return f"{seconds}s"


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_loss_curve(train_losses, val_losses, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, marker="o", label="train loss")
    plt.plot(val_losses, marker="o", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def moving_average_slope(values, window=3):
    if len(values) < 2:
        return None
    w = min(window, len(values))
    ma = np.convolve(values, np.ones(w) / w, mode="valid")
    if len(ma) < 2:
        return None
    x = np.arange(len(ma))
    return float(np.polyfit(x, ma, 1)[0])


def maybe_limit_subset(dataset, max_samples, seed):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()[:max_samples]
    return Subset(dataset, indices)


def clear_and_recreate_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else d


def extract(a, t, x_shape):
    out = a.gather(0, t)
    return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


def group_norm(channels):
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_dim=256):
        super().__init__()
        self.time_net = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down1 = ResBlock(base_ch, base_ch, time_dim)
        self.ds1 = Downsample(base_ch)
        self.down2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.ds2 = Downsample(base_ch * 2)
        self.mid = ResBlock(base_ch * 2, base_ch * 2, time_dim)
        self.us1 = Upsample(base_ch * 2)
        self.up1 = ResBlock(base_ch * 4, base_ch, time_dim)
        self.us2 = Upsample(base_ch)
        self.up2 = ResBlock(base_ch * 2, base_ch, time_dim)
        self.out_norm = group_norm(base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_net(t)
        x0 = self.in_conv(x)
        x1 = self.down1(x0, t_emb)
        x = self.ds1(x1)
        x2 = self.down2(x, t_emb)
        x = self.ds2(x2)
        x = self.mid(x, t_emb)
        x = self.us1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x, t_emb)
        x = self.us2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x, t_emb)
        return self.out_conv(F.silu(self.out_norm(x)))


class PinkNoiseOperator(nn.Module):
    def __init__(self, image_size, alpha=1.0, kappa0=None, normalize=True, eps=1e-6):
        super().__init__()
        self.image_size = int(image_size)
        self.alpha = float(alpha)
        self.kappa0 = float(kappa0) if kappa0 is not None else 1.0 / float(image_size)
        self.normalize = bool(normalize)
        self.eps = float(eps)

        fy = torch.fft.fftfreq(self.image_size).view(self.image_size, 1)
        fx = torch.fft.fftfreq(self.image_size).view(1, self.image_size)
        radius = torch.sqrt(fx * fx + fy * fy)

        power_spectrum = 1.0 / torch.pow(radius + self.kappa0, self.alpha)
        amplitude_filter = torch.sqrt(power_spectrum)

        amplitude_filter = amplitude_filter / torch.sqrt(torch.mean(amplitude_filter ** 2)).clamp_min(self.eps)
        self.register_buffer("amplitude_filter", amplitude_filter.float())

    def colorize(self, xi):
        original_dtype = xi.dtype
        x = xi.float()
        filt = self.amplitude_filter.to(device=x.device, dtype=x.dtype)[None, None, :, :]
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        colored = torch.fft.ifft2(x_fft * filt, dim=(-2, -1)).real

        if self.normalize:
            mean = colored.mean(dim=(-2, -1), keepdim=True)
            std = colored.std(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
            colored = (colored - mean) / std

        return colored.to(dtype=original_dtype)


class DDPMPink(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=2e-2,
                 image_size=64, pink_alpha=1.0, pink_kappa0=None, pink_normalize=True):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.noise_operator = PinkNoiseOperator(
            image_size=image_size,
            alpha=pink_alpha,
            kappa0=pink_kappa0,
            normalize=pink_normalize,
        )

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    def q_sample(self, x0, t, xi=None):
        xi = default(xi, torch.randn_like(x0))
        eps_pink = self.noise_operator.colorize(xi)
        return extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        ) * eps_pink

    def loss(self, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        xi = torch.randn_like(x0)
        eps_pink = self.noise_operator.colorize(xi)
        xt = extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        ) * eps_pink

        pred_eps_pink = self.model(xt, t)
        return F.mse_loss(pred_eps_pink, eps_pink)

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        beta_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_ac_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t, x.shape)

        eps_pink_theta = self.model(x, t)
        model_mean = sqrt_recip_alpha_t * (x - beta_t * eps_pink_theta / sqrt_one_minus_ac_t)

        if t_index == 0:
            return model_mean

        posterior_var_t = extract(self.posterior_variance, t, x.shape)
        z_pink = self.noise_operator.colorize(torch.randn_like(x))
        return model_mean + torch.sqrt(posterior_var_t) * z_pink

    @torch.no_grad()
    def sample_ddpm(self, shape, device):
        x = self.noise_operator.colorize(torch.randn(shape, device=device))
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_ddim(self, shape, device, sample_steps=100):
        x = self.noise_operator.colorize(torch.randn(shape, device=device))
        schedule = torch.linspace(self.timesteps - 1, 0, sample_steps, device=device).long()
        schedule = torch.unique_consecutive(schedule)

        for i, t_now in enumerate(schedule):
            t_int = int(t_now.item())
            t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
            eps_pink = self.model(x, t)

            alpha_bar_t = self.alphas_cumprod[t_int]
            if i == len(schedule) - 1:
                alpha_bar_prev = torch.tensor(1.0, device=device)
            else:
                prev_t_int = int(schedule[i + 1].item())
                alpha_bar_prev = self.alphas_cumprod[prev_t_int]

            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * eps_pink) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-1, 1)
            x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps_pink

        return x.clamp(-1, 1)


def build_transforms(image_size):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    eval_save_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    return train_tf, val_tf, eval_save_tf


def detect_fractal_layout(root):
    train_dir = root / "train"
    val_dir = root / "val"
    if train_dir.exists() and val_dir.exists():
        return "separate_train_val"
    return "single_root_classes"


def build_datasets():
    train_tf, val_tf, eval_save_tf = build_transforms(image_size)
    dataset_info = {}

    if dataset_name == "imagenet1k":
        train_dataset = datasets.ImageFolder(str(imagenet_train_dir), transform=train_tf)
        val_dataset = datasets.ImageFolder(str(imagenet_val_dir), transform=val_tf)
        real_eval_dataset = None

        train_dataset = maybe_limit_subset(train_dataset, max_train_samples, seed)
        val_dataset = maybe_limit_subset(val_dataset, max_val_samples, seed)

        dataset_info["root"] = str(imagenet_root)
        dataset_info["train_dir"] = str(imagenet_train_dir)
        dataset_info["val_dir"] = str(imagenet_val_dir)
        dataset_info["classes"] = len(train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes)
        dataset_info["split_mode"] = "predefined_train_val"
        return train_dataset, val_dataset, real_eval_dataset, dataset_info

    if dataset_name == "fractaldb1k":
        fractal_root = fractaldb1k_root
    else:
        fractal_root = fractaldb60_root

    layout = detect_fractal_layout(fractal_root)

    if layout == "separate_train_val":
        train_dir = fractal_root / "train"
        val_dir = fractal_root / "val"
        train_dataset = datasets.ImageFolder(str(train_dir), transform=train_tf)
        val_dataset = datasets.ImageFolder(str(val_dir), transform=val_tf)
        real_eval_dataset = datasets.ImageFolder(str(val_dir), transform=eval_save_tf)
        train_dataset = maybe_limit_subset(train_dataset, max_train_samples, seed)
        val_dataset = maybe_limit_subset(val_dataset, max_val_samples, seed)
        if max_real_eval_images is not None:
            real_eval_dataset = maybe_limit_subset(real_eval_dataset, max_real_eval_images, seed)
        dataset_info["split_mode"] = "predefined_train_val"
        dataset_info["train_dir"] = str(train_dir)
        dataset_info["val_dir"] = str(val_dir)
        dataset_info["classes"] = len(train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes)
        dataset_info["root"] = str(fractal_root)
        return train_dataset, val_dataset, real_eval_dataset, dataset_info

    train_full = datasets.ImageFolder(str(fractal_root), transform=train_tf)
    val_full = datasets.ImageFolder(str(fractal_root), transform=val_tf)
    eval_full = datasets.ImageFolder(str(fractal_root), transform=eval_save_tf)

    total = len(train_full)
    val_size = int(total * val_ratio)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    if max_train_samples is not None:
        train_indices = train_indices[: min(max_train_samples, len(train_indices))]
    if max_val_samples is not None:
        val_indices = val_indices[: min(max_val_samples, len(val_indices))]
    real_eval_indices = val_indices
    if max_real_eval_images is not None:
        real_eval_indices = real_eval_indices[: min(max_real_eval_images, len(real_eval_indices))]

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)
    real_eval_dataset = Subset(eval_full, real_eval_indices)

    dataset_info["split_mode"] = "random_split_from_single_root"
    dataset_info["root"] = str(fractal_root)
    dataset_info["classes"] = len(train_full.classes)
    dataset_info["total_images"] = total
    dataset_info["train_images"] = len(train_dataset)
    dataset_info["val_images"] = len(val_dataset)
    dataset_info["val_ratio"] = val_ratio
    return train_dataset, val_dataset, real_eval_dataset, dataset_info


def make_loader(dataset, batch_size, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )


@torch.no_grad()
def evaluate_loss(ddpm, loader, device):
    ddpm.eval()
    val_loss_sum = 0.0
    val_batches = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with amp_autocast():
            loss = ddpm.loss(x)
        val_loss_sum += loss.item()
        val_batches += 1
    return val_loss_sum / max(val_batches, 1)


def save_checkpoint(path, state):
    torch.save(state, path)


def load_checkpoint_if_needed(ddpm, optimizer, scaler, train_rows):
    checkpoint_path = checkpoint_dir / "last_checkpoint.pt"
    if not auto_resume_last or not checkpoint_path.exists():
        return 0, float("inf"), [], [], []

    ckpt = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    train_losses = ckpt.get("train_epoch_losses", [])
    val_losses = ckpt.get("val_epoch_losses", [])
    epoch_times = ckpt.get("epoch_times", [])
    loaded_rows = ckpt.get("train_rows", [])
    train_rows.extend(loaded_rows)
    return start_epoch, best_val_loss, train_losses, val_losses, epoch_times


def save_preview_grid(ddpm, epoch_idx, sample_steps):
    ddpm.eval()
    with torch.no_grad():
        with amp_autocast():
            samples = ddpm.sample_ddim((preview_num_images, in_channels, image_size, image_size), device, sample_steps=sample_steps)
        samples = (samples + 1.0) * 0.5
        nrow = int(math.sqrt(preview_num_images)) if preview_num_images >= 4 else preview_num_images
        grid = make_grid(samples, nrow=nrow)
        save_image(grid, preview_dir / f"preview_epoch_{epoch_idx:03d}_steps_{sample_steps}.png")


def build_real_eval_folder(real_eval_dataset, target_dir, num_images):
    clear_and_recreate_dir(target_dir)
    loader = make_loader(real_eval_dataset, batch_size=gen_batch_size, shuffle=False, drop_last=False)
    saved = 0
    for x, _ in tqdm(loader, desc="real eval images", leave=True):
        x = x.clamp(0.0, 1.0)
        b = x.size(0)
        for i in range(b):
            save_image(x[i], target_dir / f"real_{saved:06d}.png")
            saved += 1
            if saved >= num_images:
                return


def generate_images_to_folder(ddpm, out_dir, num_images, sample_steps):
    clear_and_recreate_dir(out_dir)
    saved = 0
    gen_start = time.time()
    while saved < num_images:
        current_bs = min(gen_batch_size, num_images - saved)
        with torch.no_grad():
            with amp_autocast():
                samples = ddpm.sample_ddim((current_bs, in_channels, image_size, image_size), device, sample_steps=sample_steps)
            samples = (samples + 1.0) * 0.5
        for i in range(current_bs):
            save_image(samples[i], out_dir / f"gen_{saved:06d}.png")
            saved += 1
    gen_time = time.time() - gen_start
    return saved, gen_time


def compute_generation_metrics(real_dir, gen_dir):
    metrics = torch_fidelity.calculate_metrics(
        input1=str(real_dir),
        input2=str(gen_dir),
        cuda=torch.cuda.is_available(),
        fid=enable_fid,
        kid=enable_kid,
        prc=enable_precision_recall,
        batch_size=torch_fidelity_batch_size,
        kid_subsets=kid_subsets,
        kid_subset_size=kid_subset_size,
        save_cpu_ram=True,
        verbose=False,
    )
    return metrics


def configure_stage(_stage_name, _dataset_name, _epochs):
    global stage_name, dataset_name, run_name, run_dir, checkpoint_dir, logs_dir, samples_dir, metrics_dir, preview_dir
    global train_only, enable_preview_samples, enable_generation_eval, enable_fid, enable_kid, enable_precision_recall
    global epochs

    stage_name = _stage_name
    dataset_name = _dataset_name
    epochs = int(_epochs)

    if _stage_name == "pretrain_imagenet":
        run_name = f"{noise_name}_imagenet1k_pretrain"
    elif _stage_name.startswith("finetune"):
        run_name = f"{noise_name}_{_dataset_name}_finetune_from_imagenet"
    else:
        run_name = f"{noise_name}_{_dataset_name}_{_stage_name}"

    run_dir = results_root / run_name
    checkpoint_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    samples_dir = run_dir / "samples"
    metrics_dir = run_dir / "metrics"
    preview_dir = samples_dir / "preview"

    train_only = _dataset_name == "imagenet1k"
    enable_preview_samples = not train_only
    enable_generation_eval = not train_only
    enable_fid = not train_only
    enable_kid = not train_only
    enable_precision_recall = not train_only


def stage_best_checkpoint_path(stage_name, dataset_name):
    if stage_name == "pretrain_imagenet":
        rn = f"{noise_name}_imagenet1k_pretrain"
    elif stage_name.startswith("finetune"):
        rn = f"{noise_name}_{dataset_name}_finetune_from_imagenet"
    else:
        rn = f"{noise_name}_{dataset_name}_{stage_name}"
    return results_root / rn / "checkpoints" / "best_checkpoint.pt"


def load_model_weights_only(ddpm, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(ckpt["model_state_dict"])


def make_run_config(dataset_info):
    return {
        "stage_name": stage_name,
        "noise_name": noise_name,
        "dataset_name": dataset_name,
        "device": device,
        "use_mixed_precision": use_mixed_precision,
        "mixed_precision_dtype": mixed_precision_dtype,
        "allow_tf32": allow_tf32,
        "project_dir": str(project_dir),
        "results_dir": str(run_dir),
        "dataset_info": dataset_info,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "timesteps": train_timesteps,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "pink_alpha": pink_alpha,
        "pink_kappa0": pink_kappa0,
        "pink_normalize": pink_normalize,
        "base_channels": base_channels,
        "time_embed_dim": time_embed_dim,
        "grad_clip_norm": grad_clip_norm,
        "preview_every_epochs": preview_every_epochs,
        "preview_num_images": preview_num_images,
        "preview_sample_steps": preview_sample_steps,
        "enable_generation_eval": enable_generation_eval,
        "gen_eval_steps": gen_eval_steps,
        "num_eval_gen_images": num_eval_gen_images,
        "num_eval_real_images": num_eval_real_images,
        "auto_resume_last": auto_resume_last,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "max_real_eval_images": max_real_eval_images,
        "max_gen_eval_images": max_gen_eval_images,
        "seed": seed,
        "train_only": train_only,
        "load_imagenet_pretrain_for_fractal": load_imagenet_pretrain_for_fractal,
    }


def run_stage(stage_name, dataset_name, epochs, pretrained_checkpoint=None):
    configure_stage(stage_name, dataset_name, epochs)

    for path in [project_dir, data_root, results_root, run_dir, checkpoint_dir, logs_dir, samples_dir, metrics_dir, preview_dir]:
        ensure_dir(path)

    train_dataset, val_dataset, real_eval_dataset, dataset_info = build_datasets()
    train_loader = make_loader(train_dataset, batch_size, shuffle=True, drop_last=True)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False, drop_last=False)

    run_config = make_run_config(dataset_info)
    if pretrained_checkpoint is not None:
        run_config["pretrained_checkpoint"] = str(pretrained_checkpoint)
    save_json(run_dir / "config.json", run_config)

    net = TinyUNet(in_ch=in_channels, base_ch=base_channels, time_dim=time_embed_dim)
    ddpm = DDPMPink(
        net,
        timesteps=train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        image_size=image_size,
        pink_alpha=pink_alpha,
        pink_kappa0=pink_kappa0,
        pink_normalize=pink_normalize,
    ).to(device)

    last_checkpoint_path = checkpoint_dir / "last_checkpoint.pt"
    should_resume_stage = auto_resume_last and last_checkpoint_path.exists()
    if pretrained_checkpoint is not None and load_imagenet_pretrain_for_fractal and not should_resume_stage:
        load_model_weights_only(ddpm, pretrained_checkpoint)

    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = make_grad_scaler()

    train_rows = []
    start_epoch, best_val_loss, train_epoch_losses, val_epoch_losses, epoch_times = load_checkpoint_if_needed(
        ddpm, optimizer, scaler, train_rows
    )

    total_start = time.time()

    for epoch in range(start_epoch + 1, epochs + 1):
        ddpm.train()
        epoch_start = time.time()
        train_loss_sum = 0.0
        train_batches = 0
        train_seen_images = 0

        pbar = tqdm(train_loader, desc=f"{stage_name} epoch {epoch}/{epochs}", leave=True)
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_autocast():
                loss = ddpm.loss(x)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(ddpm.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.item()
            train_loss_sum += loss_value
            train_batches += 1
            train_seen_images += x.size(0)
            pbar.set_postfix(loss=f"{loss_value:.4f}")

        if train_batches == 0:
            break

        train_mean_loss = train_loss_sum / max(train_batches, 1)
        val_mean_loss = evaluate_loss(ddpm, val_loader, device)
        epoch_time = time.time() - epoch_start
        images_per_second = train_seen_images / max(epoch_time, 1e-8)
        eta_sec = (epochs - epoch) * (sum(epoch_times + [epoch_time]) / (len(epoch_times) + 1))

        train_epoch_losses.append(train_mean_loss)
        val_epoch_losses.append(val_mean_loss)
        epoch_times.append(epoch_time)

        train_row = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_mean_loss,
            "val_loss": val_mean_loss,
            "epoch_time_sec": epoch_time,
            "images_per_second": images_per_second,
            "elapsed_stage_sec": time.time() - total_start,
            "eta_sec": eta_sec,
        }
        train_rows.append(train_row)

        checkpoint = {
            "stage_name": stage_name,
            "epoch": epoch,
            "model_state_dict": ddpm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_epoch_losses": train_epoch_losses,
            "val_epoch_losses": val_epoch_losses,
            "epoch_times": epoch_times,
            "train_rows": train_rows,
            "config": run_config,
        }

        if save_last_every_epoch:
            save_checkpoint(checkpoint_dir / "last_checkpoint.pt", checkpoint)

        if save_best_by_val_loss and val_mean_loss < best_val_loss:
            best_val_loss = val_mean_loss
            checkpoint["best_val_loss"] = best_val_loss
            save_checkpoint(checkpoint_dir / "best_checkpoint.pt", checkpoint)

        write_csv(logs_dir / "train_log.csv", list(train_rows[0].keys()), train_rows)
        plot_loss_curve(train_epoch_losses, val_epoch_losses, run_dir / "loss_curve.png")

        if enable_preview_samples and (epoch % preview_every_epochs == 0 or epoch == 1 or epoch == epochs):
            save_preview_grid(ddpm, epoch_idx=epoch, sample_steps=preview_sample_steps)

    total_train_time = time.time() - total_start

    metrics_rows = []
    real_eval_dir = metrics_dir / f"real_eval_{num_eval_real_images}"

    if enable_generation_eval:
        real_images_to_use = num_eval_real_images if max_real_eval_images is None else min(num_eval_real_images, max_real_eval_images)
        gen_images_to_use = num_eval_gen_images if max_gen_eval_images is None else min(num_eval_gen_images, max_gen_eval_images)

        build_real_eval_folder(real_eval_dataset, real_eval_dir, real_images_to_use)

        for step_count in gen_eval_steps:
            gen_dir = metrics_dir / f"generated_steps_{step_count}"
            saved_count, gen_time = generate_images_to_folder(
                ddpm=ddpm,
                out_dir=gen_dir,
                num_images=gen_images_to_use,
                sample_steps=step_count,
            )
            if saved_count == 0:
                break

            metric_start = time.time()
            metric_values = compute_generation_metrics(real_eval_dir, gen_dir)
            metric_time = time.time() - metric_start

            row = {
                "stage": stage_name,
                "sample_steps": step_count,
                "num_real_images": real_images_to_use,
                "num_generated_images": saved_count,
                "generation_time_sec": gen_time,
                "generation_speed_img_sec": saved_count / max(gen_time, 1e-8),
                "metrics_time_sec": metric_time,
                "fid": metric_values.get("frechet_inception_distance"),
                "kid_mean": metric_values.get("kernel_inception_distance_mean"),
                "kid_std": metric_values.get("kernel_inception_distance_std"),
                "precision": metric_values.get("precision"),
                "recall": metric_values.get("recall"),
            }
            metrics_rows.append(row)
            write_csv(logs_dir / "generation_metrics.csv", list(row.keys()), metrics_rows)

    summary = {
        "stage_name": stage_name,
        "run_name": run_name,
        "dataset_name": dataset_name,
        "noise_name": noise_name,
        "device": device,
        "use_mixed_precision": use_mixed_precision,
        "mixed_precision_dtype": mixed_precision_dtype,
        "allow_tf32": allow_tf32,
        "best_val_loss": best_val_loss,
        "last_train_loss": train_epoch_losses[-1] if train_epoch_losses else None,
        "last_val_loss": val_epoch_losses[-1] if val_epoch_losses else None,
        "train_loss_slope_ma3": moving_average_slope(train_epoch_losses, window=3),
        "val_loss_slope_ma3": moving_average_slope(val_epoch_losses, window=3),
        "epochs_completed": len(train_epoch_losses),
        "stage_train_time_sec": total_train_time,
        "stage_train_time_human": format_seconds(total_train_time),
        "paths": {
            "run_dir": str(run_dir),
            "checkpoints": str(checkpoint_dir),
            "logs": str(logs_dir),
            "samples": str(samples_dir),
            "metrics": str(metrics_dir),
        },
        "generation_metrics": metrics_rows,
    }
    save_json(run_dir / "summary.json", summary)

    return summary


def main():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    set_seed(seed)

    all_start = time.time()
    stage_summaries = []

    pretrain_summary = run_stage(
        stage_name="pretrain_imagenet",
        dataset_name=pretrain_dataset_name,
        epochs=pretrain_epochs,
        pretrained_checkpoint=None,
    )
    stage_summaries.append(pretrain_summary)

    imagenet_best = stage_best_checkpoint_path("pretrain_imagenet", pretrain_dataset_name)

    finetune_summary = run_stage(
        stage_name=f"finetune_{finetune_dataset_name}",
        dataset_name=finetune_dataset_name,
        epochs=finetune_epochs,
        pretrained_checkpoint=imagenet_best,
    )
    stage_summaries.append(finetune_summary)

    combined_summary = {
        "noise_name": noise_name,
        "pretrain_dataset": pretrain_dataset_name,
        "finetune_dataset": finetune_dataset_name,
        "total_elapsed_sec": time.time() - all_start,
        "total_elapsed_human": format_seconds(time.time() - all_start),
        "stage_summaries": stage_summaries,
    }
    ensure_dir(results_root)
    save_json(results_root / f"{noise_name}_two_stage_summary.json", combined_summary)


main()
