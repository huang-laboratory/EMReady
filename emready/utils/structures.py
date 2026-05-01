"""Structure-mask helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import MMCIFParser, PDBParser
from numba import njit
import warnings


def load_heavy_atom_coords(structure_file: str | Path) -> np.ndarray:
    structure_file = Path(structure_file)
    suffix = structure_file.suffix.lower()
    if suffix in {".pdb", ".pdb1"}:
        parser = PDBParser(QUIET=True)
    elif suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise RuntimeError(f"Unknown structure file type: {structure_file}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)
        structures = parser.get_structure("mask", structure_file)

    coords = []
    for atom in structures[0].get_atoms():
        if atom.element == "H":
            continue
        coords.append(atom.get_coord())
    return np.asarray(coords, dtype=np.float32)


def build_structure_mask(
    atoms: np.ndarray,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    nxyz: np.ndarray,
    radius: float,
) -> np.ndarray:
    atom_shifted = atoms - origin
    lower = np.floor((np.min(atom_shifted, axis=0) - radius) / voxel_size).astype(np.int64)
    upper = np.ceil((np.max(atom_shifted, axis=0) + radius) / voxel_size).astype(np.int64)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, nxyz - 1)

    if np.any(lower > upper):
        return np.zeros(nxyz[::-1], dtype=np.int16)

    local_start = lower
    local_shape = (upper - lower + 1).astype(np.int64)
    local_atoms = atom_shifted - local_start.astype(np.float32) * voxel_size
    local_mask = _build_structure_mask_local(
        local_atoms.astype(np.float32),
        voxel_size.astype(np.float32),
        local_shape.astype(np.int64),
        float(radius),
    )

    mask = np.zeros(nxyz[::-1], dtype=np.int16)
    mask[
        local_start[2] : local_start[2] + local_shape[2],
        local_start[1] : local_start[1] + local_shape[1],
        local_start[0] : local_start[0] + local_shape[0],
    ] = local_mask
    return mask


@njit(cache=True)
def _build_structure_mask_local(
    atoms_local: np.ndarray,
    voxel_size: np.ndarray,
    local_shape: np.ndarray,
    radius: float,
) -> np.ndarray:
    radius2 = radius * radius
    mask = np.zeros((local_shape[2], local_shape[1], local_shape[0]), dtype=np.int16)
    vx, vy, vz = voxel_size[0], voxel_size[1], voxel_size[2]

    for i in range(atoms_local.shape[0]):
        atom = atoms_local[i]
        lower = np.floor((atom - radius) / voxel_size).astype(np.int64)
        upper = np.ceil((atom + radius) / voxel_size).astype(np.int64)

        x_start = max(lower[0], 0)
        x_end = min(upper[0], local_shape[0] - 1)
        y_start = max(lower[1], 0)
        y_end = min(upper[1], local_shape[1] - 1)
        z_start = max(lower[2], 0)
        z_end = min(upper[2], local_shape[2] - 1)

        for x in range(x_start, x_end + 1):
            dx = x * vx - atom[0]
            dx2 = dx * dx
            for y in range(y_start, y_end + 1):
                dy = y * vy - atom[1]
                dy2 = dy * dy
                for z in range(z_start, z_end + 1):
                    if mask[z, y, x] != 0:
                        continue
                    dz = z * vz - atom[2]
                    distance2 = dx2 + dy2 + dz * dz
                    if distance2 < radius2:
                        mask[z, y, x] = 1
    return mask
