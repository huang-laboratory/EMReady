"""Main-chain map preprocessing and tiled inference helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import mrcfile
import numpy as np
import torch

from emready.models.bimcunet_mainchain import BiMCUnetMainTask
from emready.utils.checkpoints import load_state_dict_file


MAIN_CHAIN_MODEL_KWARGS = {
    "in_channels": 1,
    "out_channels": 4,
    "base_dim": 32,
    "patch_size": 4,
    "block_config": [2, 2, 2, 2, 2, 2, 2],
}


def load_mrc_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with mrcfile.open(path, mode="r") as handle:
        data = np.asarray(handle.data.copy(), dtype=np.float32)
        voxel_size = np.asarray(
            [handle.voxel_size.x, handle.voxel_size.y, handle.voxel_size.z],
            dtype=np.float32,
        )
        nxyzstart = np.asarray(
            [handle.header.nxstart, handle.header.nystart, handle.header.nzstart],
            dtype=np.int64,
        )
        origin = np.asarray(
            [handle.header.origin.x, handle.header.origin.y, handle.header.origin.z],
            dtype=np.float32,
        )
    return data, voxel_size, nxyzstart, origin


def write_mrc_volume(
    path: str | Path,
    data_zyx: np.ndarray,
    voxel_size_xyz: np.ndarray,
    nxyzstart_xyz: np.ndarray,
    origin_xyz: np.ndarray,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(out_path), overwrite=True) as handle:
        handle.set_data(data_zyx.astype(np.float32, copy=False))
        handle.voxel_size = tuple(float(v) for v in voxel_size_xyz.tolist())
        handle.header.nxstart = int(nxyzstart_xyz[0])
        handle.header.nystart = int(nxyzstart_xyz[1])
        handle.header.nzstart = int(nxyzstart_xyz[2])
        handle.header.origin.x = float(origin_xyz[0])
        handle.header.origin.y = float(origin_xyz[1])
        handle.header.origin.z = float(origin_xyz[2])


def normalize_exp_map(exp_map: np.ndarray, percentile: float) -> np.ndarray:
    """Clip to positive percentile then z-score (matches emalign stage1 prep)."""
    positive = exp_map[exp_map > 0]
    if positive.size == 0:
        print("# Warning: exp map has no positive voxels; returning zeros.")
        return np.zeros_like(exp_map, dtype=np.float32)
    vmax = float(np.percentile(positive, percentile))
    if vmax <= 0:
        return np.zeros_like(exp_map, dtype=np.float32)
    clipped = np.clip(exp_map, a_min=0.0, a_max=vmax).astype(np.float32, copy=False)
    mean = float(clipped.mean())
    std = float(clipped.std())
    print(f"# exp_map mean/std before z-score: {mean:.6f} / {std:.6f}")
    if std < 1e-8:
        print("# Warning: exp_map std is too small; returning zeros.")
        return np.zeros_like(clipped, dtype=np.float32)
    return ((clipped - mean) / std).astype(np.float32, copy=False)


def compute_padded_dim(size: int, box_size: int, stride: int) -> int:
    if size <= box_size:
        return box_size
    return box_size + math.ceil((size - box_size) / stride) * stride


def pad_axis(size: int, box_size: int, stride: int, pad_size: int) -> tuple[int, int, int]:
    inner = int(size) + 2 * int(pad_size)
    target = compute_padded_dim(inner, box_size, stride)
    extra = target - inner
    pad_before = int(pad_size) + extra // 2
    pad_after = int(pad_size) + (extra - extra // 2)
    return target, pad_before, pad_after


def pad_map_for_tiling(
    map_zyx: np.ndarray,
    box_size: int,
    stride: int,
    pad_size: int | None = None,
) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int]]:
    depth, height, width = map_zyx.shape
    if pad_size is None:
        pad_size = box_size // 2
    padded_z, z0, _ = pad_axis(depth, box_size, stride, pad_size)
    padded_y, y0, _ = pad_axis(height, box_size, stride, pad_size)
    padded_x, x0, _ = pad_axis(width, box_size, stride, pad_size)
    padded = np.zeros((padded_z, padded_y, padded_x), dtype=np.float32)
    padded[z0 : z0 + depth, y0 : y0 + height, x0 : x0 + width] = map_zyx
    return padded, (depth, height, width), (z0, y0, x0)


def sliding_starts(size: int, box_size: int, stride: int) -> Iterable[int]:
    if size <= box_size:
        yield 0
        return
    end = size - box_size
    value = 0
    while value <= end:
        yield value
        value += stride


def make_gaussian_weight_kernel(
    box_size: int,
    *,
    min_weight: float = 1.0,
    max_weight: float = 3.0,
    sigma_scale: float = 0.5,
) -> np.ndarray:
    coords = np.linspace(-1.0, 1.0, box_size, dtype=np.float32)
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    radius2 = x * x + y * y + z * z
    kernel = np.exp(-0.5 * radius2 / (sigma_scale * sigma_scale)).astype(np.float32)
    kmin = float(kernel.min())
    kmax = float(kernel.max())
    if kmax == kmin:
        return np.full((box_size, box_size, box_size), max_weight, dtype=np.float32)
    kernel = (kernel - kmin) / (kmax - kmin)
    kernel = kernel * (max_weight - min_weight) + min_weight
    return kernel.astype(np.float32, copy=False)


def crop_to_original(
    volume: np.ndarray,
    original_shape: tuple[int, int, int],
    crop_start: tuple[int, int, int],
) -> np.ndarray:
    depth, height, width = original_shape
    z0, y0, x0 = crop_start
    crop = (slice(z0, z0 + depth), slice(y0, y0 + height), slice(x0, x0 + width))
    if volume.ndim == 3:
        return volume[crop].astype(np.float32, copy=False)
    return volume[(slice(None), *crop)].astype(np.float32, copy=False)


def load_main_chain_model(weight_file: Path, device: torch.device) -> BiMCUnetMainTask:
    print(f"# Loading main-chain model weights from {weight_file}")
    model = BiMCUnetMainTask(**MAIN_CHAIN_MODEL_KWARGS)
    state_dict = load_state_dict_file(weight_file, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def infer_main_chain(
    model: torch.nn.Module,
    exp_map_zyx: np.ndarray,
    *,
    box_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    weight_kernel: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_mc, pred_class_idx) cropped to original shape."""
    padded_map, original_shape, crop_start = pad_map_for_tiling(
        exp_map_zyx,
        box_size=box_size,
        stride=stride,
    )
    depth, height, width = padded_map.shape

    logits_sum = np.zeros((3, depth, height, width), dtype=np.float32)
    mc_sum = np.zeros((depth, height, width), dtype=np.float32)
    weight_sum = np.zeros((depth, height, width), dtype=np.float32)

    starts_z = list(sliding_starts(depth, box_size, stride))
    starts_y = list(sliding_starts(height, box_size, stride))
    starts_x = list(sliding_starts(width, box_size, stride))
    total_tiles = len(starts_z) * len(starts_y) * len(starts_x)
    print(f"# Inference tiles: {total_tiles} ({len(starts_z)}x{len(starts_y)}x{len(starts_x)})")

    batch_chunks: list[np.ndarray] = []
    batch_pos: list[tuple[int, int, int]] = []

    def flush_batch() -> None:
        if not batch_chunks:
            return
        x = torch.from_numpy(np.stack(batch_chunks, axis=0)).unsqueeze(1).to(
            device=device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            out = model(x)
            mc = out["mc"].squeeze(1).detach().cpu().numpy().astype(np.float32, copy=False)
            logits = out["class_logits"].detach().cpu().numpy().astype(np.float32, copy=False)
        for idx, (z0, y0, x0) in enumerate(batch_pos):
            target = (
                slice(z0, z0 + box_size),
                slice(y0, y0 + box_size),
                slice(x0, x0 + box_size),
            )
            if weight_kernel is None:
                logits_sum[:, target[0], target[1], target[2]] += logits[idx]
                mc_sum[target] += mc[idx]
                weight_sum[target] += 1.0
            else:
                logits_sum[:, target[0], target[1], target[2]] += logits[idx] * weight_kernel[None, ...]
                mc_sum[target] += mc[idx] * weight_kernel
                weight_sum[target] += weight_kernel
        batch_chunks.clear()
        batch_pos.clear()

    for z0 in starts_z:
        for y0 in starts_y:
            for x0 in starts_x:
                batch_chunks.append(
                    padded_map[z0 : z0 + box_size, y0 : y0 + box_size, x0 : x0 + box_size]
                )
                batch_pos.append((z0, y0, x0))
                if len(batch_chunks) >= batch_size:
                    flush_batch()
    flush_batch()

    logits_avg = logits_sum / np.clip(weight_sum[None, ...], a_min=1e-6, a_max=None)
    mc_avg = mc_sum / np.clip(weight_sum, a_min=1e-6, a_max=None)
    class_idx = np.argmax(logits_avg, axis=0).astype(np.float32, copy=False)

    pred_mc = crop_to_original(mc_avg, original_shape, crop_start)
    pred_class = crop_to_original(class_idx, original_shape, crop_start)
    return pred_mc, pred_class
