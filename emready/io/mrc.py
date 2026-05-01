"""MRC map loading, interpolation, and writing helpers."""

from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np

from emready.utils.interp3d import Interp3D


_INTERP3D = Interp3D()


def parse_map(
    map_file: str | Path,
    ignorestart: bool,
    apix: float | None = None,
    origin_shift: np.ndarray | None = None,
):
    """Read an MRC map and optionally resample it to an isotropic grid."""
    with mrcfile.open(map_file, mode="r") as mrc:
        volume = np.asarray(mrc.data.copy(), dtype=np.float32)
        voxel_size = np.asarray(
            [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z],
            dtype=np.float32,
        )
        ncrsstart = np.asarray(
            [mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart],
            dtype=np.float32,
        )
        origin = np.asarray(
            [mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z],
            dtype=np.float32,
        )
        ncrs = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
        angle = np.asarray(
            [mrc.header.cellb.alpha, mrc.header.cellb.beta, mrc.header.cellb.gamma],
            dtype=np.float32,
        )
        mapcrs = np.subtract([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], 1)

    if not np.all(angle == 90.0):
        raise ValueError("Input grid is not orthogonal.")

    order = np.asarray([0, 1, 2], dtype=np.int64)
    for i in range(3):
        order[mapcrs[i]] = i
    nxyzstart = np.asarray([ncrsstart[i] for i in order])
    nxyz = np.asarray([ncrs[i] for i in order])
    nxyz_origin = nxyz.copy()

    volume = np.transpose(volume, axes=2 - order[::-1])

    if not ignorestart:
        origin += np.multiply(nxyzstart, voxel_size)

    if apix is not None:
        same_grid = (
            voxel_size[0] == voxel_size[1] == voxel_size[2] == apix
            and origin_shift is None
        )
        if not same_grid:
            _INTERP3D.del_mapout()
            target_voxel_size = np.asarray([apix, apix, apix], dtype=np.float32)
            print(f"# Rescale voxel size from {voxel_size} to {target_voxel_size}")
            shift = origin_shift if origin_shift is not None else (0.0, 0.0, 0.0)
            _INTERP3D.cubic(
                volume,
                voxel_size[2],
                voxel_size[1],
                voxel_size[0],
                apix,
                shift[2],
                shift[1],
                shift[0],
                nxyz[2],
                nxyz[1],
                nxyz[0],
            )
            if origin_shift is not None:
                origin += origin_shift
            volume = _INTERP3D.mapout
            nxyz = np.asarray(
                [_INTERP3D.pextx, _INTERP3D.pexty, _INTERP3D.pextz],
                dtype=np.int64,
            )
            voxel_size = target_voxel_size

    expected_shape = np.asarray([volume.shape[2], volume.shape[1], volume.shape[0]])
    if not np.all(nxyz == expected_shape):
        raise ValueError(f"MRC axis metadata is inconsistent with data shape: {nxyz} vs {expected_shape}")

    return volume, origin, nxyz, voxel_size, nxyz_origin


def align_origin_to_grid(map_file: str | Path, apix: float):
    volume, origin, nxyz, voxel_size, nxyz_origin = parse_map(
        map_file, ignorestart=False, apix=apix
    )
    if not origin_is_on_grid(origin, voxel_size):
        origin_shift = grid_origin_shift(origin, voxel_size)
        volume, origin, nxyz, voxel_size, nxyz_origin = parse_map(
            map_file,
            ignorestart=False,
            apix=apix,
            origin_shift=origin_shift,
        )
    if not origin_is_on_grid(origin, voxel_size):
        raise ValueError("Map origin could not be aligned to the voxel grid.")
    return volume, origin, nxyz, voxel_size, nxyz_origin


def inverse_map(
    volume: np.ndarray,
    nxyz: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    old_voxel_size: np.ndarray,
    origin_shift,
):
    _INTERP3D.del_mapout()
    _INTERP3D.inverse_cubic(
        volume,
        voxel_size[2],
        voxel_size[1],
        voxel_size[0],
        old_voxel_size[2],
        old_voxel_size[1],
        old_voxel_size[0],
        origin_shift[2],
        origin_shift[1],
        origin_shift[0],
        nxyz[2],
        nxyz[1],
        nxyz[0],
    )
    origin += origin_shift
    nxyz = np.asarray([_INTERP3D.pextx, _INTERP3D.pexty, _INTERP3D.pextz], dtype=np.int64)
    volume = _INTERP3D.mapout
    voxel_size = old_voxel_size
    return volume, origin, nxyz, voxel_size


def origin_is_on_grid(origin: np.ndarray, voxel_size: np.ndarray, tol: float = 1e-3) -> bool:
    return bool(np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < tol))


def grid_origin_shift(origin: np.ndarray, voxel_size: np.ndarray) -> np.ndarray:
    return (np.round(origin / voxel_size) - origin / voxel_size) * voxel_size


def write_map(
    file_name: str | Path,
    volume: np.ndarray,
    voxel_size: np.ndarray,
    origin=(0.0, 0.0, 0.0),
    nxyzstart=(0, 0, 0),
) -> None:
    file_name = Path(file_name)
    file_name.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(file_name, overwrite=True) as mrc:
        mrc.set_data(volume.astype(np.float32, copy=False))
        (mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart) = nxyzstart
        (mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z) = origin
        mrc.voxel_size = [voxel_size[i] for i in range(3)]

