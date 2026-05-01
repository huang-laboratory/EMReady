"""Sliding-window chunk helpers for 3D map inference."""

from __future__ import annotations

import numpy as np


def make_gaussian_weight_kernel(
    box_size: int,
    min_weight: float = 1.0,
    max_weight: float = 3.0,
    sigma_scale: float = 0.5,
) -> np.ndarray:
    """Create a 3D Gaussian patch-blending kernel normalized to [min, max]."""
    if box_size <= 0:
        raise ValueError("box_size must be positive")
    if sigma_scale <= 0:
        raise ValueError("sigma_scale must be positive")
    if max_weight <= min_weight:
        raise ValueError("max_weight must be greater than min_weight")

    coords = np.linspace(-1.0, 1.0, box_size, dtype=np.float32)
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    radius2 = x * x + y * y + z * z
    kernel = np.exp(-0.5 * radius2 / (sigma_scale * sigma_scale)).astype(np.float32)
    kernel_min = float(kernel.min())
    kernel_max = float(kernel.max())
    if kernel_max == kernel_min:
        return np.full((box_size, box_size, box_size), max_weight, dtype=np.float32)
    kernel = (kernel - kernel_min) / (kernel_max - kernel_min)
    kernel = kernel * (max_weight - min_weight) + min_weight
    return kernel.astype(np.float32, copy=False)


def pad_map(volume: np.ndarray, box_size: int, dtype=np.float32, padding: float = 0.0):
    shape = np.shape(volume)
    padded = np.full(
        (
            shape[0] + 2 * box_size,
            shape[1] + 2 * box_size,
            shape[2] + 2 * box_size,
        ),
        padding,
        dtype=dtype,
    )
    padded[
        box_size : box_size + shape[0],
        box_size : box_size + shape[1],
        box_size : box_size + shape[2],
    ] = volume
    return padded


def chunk_generator(padded_map: np.ndarray, maximum: float, box_size: int, stride: int):
    if stride > box_size:
        raise ValueError("stride must not exceed box_size")
    if maximum <= 0:
        raise ValueError("maximum must be positive")

    padded_shape = np.shape(padded_map)
    start_point = box_size - stride
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    while cur_z + stride < padded_shape[2] - box_size:
        next_chunk = padded_map[
            cur_x : cur_x + box_size,
            cur_y : cur_y + box_size,
            cur_z : cur_z + box_size,
        ]
        cur_x0, cur_y0, cur_z0 = cur_x, cur_y, cur_z
        cur_x += stride
        if cur_x + stride >= padded_shape[0] - box_size:
            cur_y += stride
            cur_x = start_point
            if cur_y + stride >= padded_shape[1] - box_size:
                cur_z += stride
                cur_y = start_point
                cur_x = start_point

        if next_chunk.max() <= 0.0:
            continue
        yield cur_x0, cur_y0, cur_z0, next_chunk.clip(min=0.0, max=maximum) / maximum


def get_batch_from_generator(generator, batch_size: int, dtype=np.float32):
    positions = []
    batch = []
    for _ in range(batch_size):
        try:
            output = next(generator)
        except StopIteration:
            break
        positions.append(output[:3])
        batch.append(output[3])
    return positions, np.asarray(batch, dtype=dtype)


def map_batch_to_map(
    pred_map: np.ndarray,
    denominator: np.ndarray,
    positions,
    batch: np.ndarray,
    box_size: int,
    weight_kernel: np.ndarray | None = None,
):
    for position, chunk in zip(positions, batch):
        target = (
            slice(position[0], position[0] + box_size),
            slice(position[1], position[1] + box_size),
            slice(position[2], position[2] + box_size),
        )
        if weight_kernel is None:
            pred_map[target] += chunk
            denominator[target] += 1
            continue

        pred_map[target] += chunk * weight_kernel
        denominator[target] += weight_kernel
    return pred_map, denominator
