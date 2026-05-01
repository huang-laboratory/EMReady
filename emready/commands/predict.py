"""EMReady2 3D map inference command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

import emready
from emready.io.mrc import (
    align_origin_to_grid,
    grid_origin_shift,
    inverse_map,
    origin_is_on_grid,
    parse_map,
    write_map,
)
from emready.models import BiMCUnet
from emready.utils.checkpoints import load_state_dict_file
from emready.utils.chunks import (
    chunk_generator,
    get_batch_from_generator,
    make_gaussian_weight_kernel,
    map_batch_to_map,
    pad_map,
)
from emready.utils.structures import build_structure_mask, load_heavy_atom_coords


BOX_SIZE = 64
PERCENTILE = 99.999
DEFAULT_MODEL_FILES = {
    1.0: ("model_1p0.pt", "EMReady2_BiMCUNet_apix1p0.pt"),
    0.6: ("model_0p6.pt", "EMReady2_BiMCUNet_apix0p6.pt"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emready",
        description=(
            "EMReady2 3D map inference command. "
            "You must provide input and output either via positional arguments "
            "or via --input/--output."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "in_map",
        nargs="?",
        help=(
            "Input EM density map in MRC2014 format (.mrc/.map). "
            "required (unless provided by --input)."
        ),
    )
    parser.add_argument(
        "out_map",
        nargs="?",
        help=(
            "Output processed density map in MRC2014 format (.mrc). "
            "required (unless provided by --output)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"EMReady {emready.__version__}")

    basic_group = parser.add_argument_group("Basic Arguments")
    basic_group.add_argument(
        "--input",
        "-bi",
        dest="in_map_opt",
        help=(
            "Input EM density map in MRC2014 format (.mrc/.map). "
            "required (unless provided by positional in_map). default=None."
        ),
    )
    basic_group.add_argument(
        "--output",
        "-bo",
        dest="out_map_opt",
        help=(
            "Output processed density map in MRC2014 format (.mrc). "
            "required (unless provided by positional out_map). default=None."
        ),
    )
    basic_group.add_argument(
        "--model_path",
        "-bmp",
        type=Path,
        default=None,
        help=(
            "Single BiMCUnet weight file (.pt). "
            "If omitted, built-in model auto-selection is used by voxel size from model_weights/. default=None."
        ),
    )
    basic_group.add_argument(
        "--stride",
        "-bs",
        type=int,
        default=16,
        help=(
            "Sliding-window stride for 64^3 patch inference. "
            "valid range: [6, 64]. default=16."
        ),
    )
    basic_group.add_argument(
        "--batch_size",
        "-bb",
        type=int,
        default=16,
        help="Batch size (number of patches per forward pass). default=16.",
    )
    basic_group.add_argument(
        "--gpu_id",
        "-bg",
        type=str,
        default="0",
        help=(
            "CUDA visible device id string. "
            "Examples: '0', '1', '0,1'. default=0."
        ),
    )
    basic_group.add_argument(
        "--blend_mode",
        "-bbm",
        choices=("uniform", "gaussian"),
        default="gaussian",
        help=(
            "Patch aggregation weighting mode. "
            "'uniform' preserves original EMReady2 behavior; 'gaussian' uses center-weighted fusion. "
            "default=gaussian."
        ),
    )
    basic_group.add_argument(
        "--gaussian_sigma_scale",
        "-bgs",
        type=float,
        default=0.5,
        help=(
            "Gaussian sigma scale for patch fusion when --blend_mode=gaussian. "
            "must be >0. default=0.5."
        ),
    )

    mask_group = parser.add_argument_group("Mask Arguments")
    mask_group.add_argument(
        "--mask_map",
        "-mm",
        type=Path,
        default=None,
        help=(
            "Mask density map (.mrc/.map). "
            "Cannot be used together with --mask_str. default=None."
        ),
    )
    mask_group.add_argument(
        "--mask_contour",
        "-mc",
        type=float,
        default=0.0,
        help=(
            "Contour threshold for binarizing --mask_map. "
            "Voxels >= threshold are kept (or removed when --inverse_mask). default=0.0."
        ),
    )
    mask_group.add_argument(
        "--mask_str",
        "-ms",
        type=Path,
        default=None,
        help=(
            "Structure file (.pdb/.pdb1/.cif) used to generate a zone mask. "
            "Cannot be used together with --mask_map. default=None."
        ),
    )
    mask_group.add_argument(
        "--mask_str_radius",
        "-mr",
        type=float,
        default=4.0,
        help="Mask radius in Angstrom when --mask_str is used. default=4.0.",
    )
    mask_group.add_argument(
        "--mask_out",
        "-mo",
        type=Path,
        default=None,
        help="Optional output path for the generated binary mask map (.mrc). default=None.",
    )
    mask_group.add_argument(
        "--inverse_mask",
        "--inverse",
        "-mi",
        action="store_true",
        default=False,
        help="Invert mask keep/remove logic for --mask_map or --mask_str. default=False.",
    )
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    args.in_map = args.in_map_opt or args.in_map
    args.out_map = args.out_map_opt or args.out_map
    if not args.in_map or not args.out_map:
        parser.error("input and output maps are required")
    args.in_map = Path(args.in_map)
    args.out_map = Path(args.out_map)
    if args.model_path is not None and args.model_path.is_dir():
        parser.error("--model_path/-bmp must be a single weight file, not a directory")
    if not (6 <= args.stride <= 64):
        parser.error("--stride/-bs must be in the range [6, 64]")
    if args.gaussian_sigma_scale <= 0:
        parser.error("--gaussian_sigma_scale must be positive")
    if args.mask_map is not None and args.mask_str is not None:
        parser.error("--mask_map/-mm and --mask_str/-ms cannot be provided at the same time")
    return args


def choose_apix_and_weight(in_map: Path, model_path: Path | None) -> tuple[float, Path]:
    _, _, _, voxel_size, _ = parse_map(in_map, ignorestart=False)
    print(f"# Voxel size of the input map: {voxel_size}")
    if voxel_size.min() >= 0.8:
        print("# Using the EMReady2 BiMCUnet model of 1.0 Angstrom grid size")
        apix = 1.0
    else:
        print("# Using the EMReady2 BiMCUnet model of 0.6 Angstrom grid size")
        apix = 0.6

    if model_path is not None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model weight file does not exist: {model_path}")
        return apix, model_path

    model_dir = Path(__file__).resolve().parents[2] / "model_weights"
    for file_name in DEFAULT_MODEL_FILES[apix]:
        candidate = model_dir / file_name
        if candidate.is_file():
            return apix, candidate
    names = ", ".join(DEFAULT_MODEL_FILES[apix])
    raise FileNotFoundError(
        f"No default model weight found for apix={apix} in {model_dir}. "
        f"Expected one of: {names}. Use --model_path/-bmp to pass a weight file."
    )


def load_model(weight_file: Path, device: torch.device) -> BiMCUnet:
    print(f"# Loading model weights from {weight_file}")
    model = BiMCUnet(
        in_nc=1,
        config=[2, 2, 2, 2, 2, 2, 2],
        dim=32,
        out_nc=1,
        patch_size=4,
    )
    state_dict = load_state_dict_file(weight_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def apply_mask_map(args, map_volume, origin, nxyz, voxel_size, nxyzstart):
    print("# Loading the mask map...")
    mask, origin_mask, nxyz_mask, voxel_size_mask, _ = align_origin_to_grid(
        args.mask_map,
        apix=float(voxel_size[0]),
    )
    nxyzstart_mask = np.round(origin_mask / voxel_size_mask).astype(np.int64)
    print(f"# Mask map dimensions: {nxyz_mask}")

    if np.any(nxyz_mask > nxyz):
        raise ValueError("Mask map dimensions cannot exceed input map dimensions.")

    if not np.all(nxyz_mask == nxyz):
        pad_mask = np.zeros(nxyz[::-1], dtype=np.float32)
        nxyz_shift = nxyzstart_mask - nxyzstart
        pad_mask[
            nxyz_shift[2] : nxyz_shift[2] + nxyz_mask[2],
            nxyz_shift[1] : nxyz_shift[1] + nxyz_mask[1],
            nxyz_shift[0] : nxyz_shift[0] + nxyz_mask[0],
        ] = mask
        mask = pad_mask
        nxyzstart_mask = nxyzstart
        voxel_size_mask = voxel_size

    if args.inverse_mask:
        masked_volume = np.where(mask < args.mask_contour, map_volume, 0).astype(np.float32)
        binary_mask = np.where(mask < args.mask_contour, 1, 0).astype(np.float32)
    else:
        masked_volume = np.where(mask >= args.mask_contour, map_volume, 0).astype(np.float32)
        binary_mask = np.where(mask >= args.mask_contour, 1, 0).astype(np.float32)

    if args.mask_out is not None:
        print(f"# Saving the binary mask map to {args.mask_out}")
        write_map(args.mask_out, binary_mask, voxel_size_mask, nxyzstart=nxyzstart_mask)
    return masked_volume


def apply_structure_mask(args, map_volume, origin, nxyz, voxel_size, nxyzstart):
    print(f"# Generating the mask map from the structure file {args.mask_str}...")
    atoms = load_heavy_atom_coords(args.mask_str)
    mask = build_structure_mask(atoms, origin, voxel_size, nxyz, args.mask_str_radius)
    if args.inverse_mask:
        mask = 1 - mask

    masked_volume = map_volume * mask.astype(np.float32)
    if args.mask_out is not None:
        print(f"# Saving the binary mask map to {args.mask_out}")
        write_map(args.mask_out, mask.astype(np.float32), voxel_size, nxyzstart=nxyzstart)
    return masked_volume


def prepare_map(args, apix: float):
    print("# Loading the input map...")
    map_volume, origin, nxyz, voxel_size, nxyz_origin = align_origin_to_grid(
        args.in_map,
        apix=apix,
    )
    nxyzstart = np.round(origin / voxel_size).astype(np.int64)
    print(f"# Original map dimensions: {nxyz_origin}")
    print(f"# Map dimensions at {apix} Angstrom grid size: {nxyz}")

    _, _, _, old_voxel_size, _ = parse_map(args.in_map, ignorestart=False, apix=None)

    if args.mask_map is not None:
        map_volume = apply_mask_map(args, map_volume, origin, nxyz, voxel_size, nxyzstart)
    if args.mask_str is not None:
        map_volume = apply_structure_mask(args, map_volume, origin, nxyz, voxel_size, nxyzstart)

    return map_volume, origin, nxyz, voxel_size, old_voxel_size, nxyzstart


def build_weight_kernel(blend_mode: str, sigma_scale: float):
    if blend_mode == "uniform":
        print("# Patch blending mode: uniform")
        return None
    if blend_mode == "gaussian":
        print(
            "# Patch blending mode: gaussian "
            f"(weight range [1, 3], sigma_scale={sigma_scale})"
        )
        return make_gaussian_weight_kernel(
            BOX_SIZE,
            min_weight=1.0,
            max_weight=3.0,
            sigma_scale=sigma_scale,
        )
    raise ValueError(f"Unknown blend mode: {blend_mode}")


def predict_volume(
    model,
    map_volume: np.ndarray,
    batch_size: int,
    stride: int,
    device: torch.device,
    blend_mode: str,
    gaussian_sigma_scale: float,
):
    positive = map_volume[map_volume > 0]
    if positive.size == 0:
        raise ValueError("Input map has no positive density after masking.")

    padded_map = pad_map(map_volume, BOX_SIZE, dtype=np.float32, padding=0.0)
    maximum = np.percentile(positive, PERCENTILE)
    if maximum <= 0:
        raise ValueError("Input map percentile normalization is not positive.")

    map_pred = np.zeros_like(padded_map, dtype=np.float32)
    denominator = np.zeros_like(padded_map, dtype=np.float32)
    generator = chunk_generator(padded_map, maximum, BOX_SIZE, stride)
    weight_kernel = build_weight_kernel(blend_mode, gaussian_sigma_scale)

    print("# Start processing...")
    with torch.inference_mode():
        while True:
            positions, chunks = get_batch_from_generator(generator, batch_size, dtype=np.float32)
            if len(positions) == 0:
                break
            x = torch.from_numpy(chunks).view(-1, 1, BOX_SIZE, BOX_SIZE, BOX_SIZE).to(device)
            y_pred = model(x).view(-1, BOX_SIZE, BOX_SIZE, BOX_SIZE)
            y_pred = y_pred.cpu().numpy() * 90
            map_pred, denominator = map_batch_to_map(
                map_pred,
                denominator,
                positions,
                y_pred,
                BOX_SIZE,
                weight_kernel=weight_kernel,
            )

    return map_pred / denominator.clip(min=1)


def restore_original_grid(map_pred, nxyz, origin, voxel_size, old_voxel_size):
    cropped = map_pred[
        BOX_SIZE : BOX_SIZE + nxyz[2],
        BOX_SIZE : BOX_SIZE + nxyz[1],
        BOX_SIZE : BOX_SIZE + nxyz[0],
    ]

    print(f"# Interpolating the voxel size from {voxel_size} back to {old_voxel_size}")
    origin = np.round(origin / voxel_size).astype(np.int64) * voxel_size
    origin_shift = [0.0, 0.0, 0.0]
    if not origin_is_on_grid(origin, old_voxel_size):
        origin_shift = grid_origin_shift(origin, old_voxel_size)
    restored, origin, nxyz, voxel_size = inverse_map(
        cropped,
        nxyz,
        origin,
        voxel_size,
        old_voxel_size,
        origin_shift,
    )
    if not origin_is_on_grid(origin, old_voxel_size):
        raise ValueError("Output origin is not aligned to the original voxel grid.")
    nxyzstart = np.round(origin / voxel_size).astype(np.int64)
    return restored, voxel_size, nxyzstart


def run(args: argparse.Namespace) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device("cuda")
    print(f"# Running on {torch.cuda.device_count()} GPU(s)")

    apix, weight_file = choose_apix_and_weight(args.in_map, args.model_path)
    model = load_model(weight_file, device=device)
    torch.cuda.empty_cache()

    map_volume, origin, nxyz, voxel_size, old_voxel_size, _ = prepare_map(args, apix)
    map_pred = predict_volume(
        model,
        map_volume,
        args.batch_size,
        args.stride,
        device,
        args.blend_mode,
        args.gaussian_sigma_scale,
    )
    restored, voxel_size, nxyzstart = restore_original_grid(
        map_pred,
        nxyz,
        origin,
        voxel_size,
        old_voxel_size,
    )

    print(
        "# Saving the processed map that has been interpolated back to "
        f"the original grid size to {args.out_map}"
    )
    write_map(args.out_map, restored, voxel_size, nxyzstart=nxyzstart)


def main(argv=None) -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args(argv), parser)
    run(args)
    return 0
