"""EMReady ligand-map inference command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

import emready
from emready.utils.chunks import make_gaussian_weight_kernel
from emready.utils.ligand_map import (
    build_ligand_binary_mask,
    build_ligand_mask,
    infer_ligand_sim_class,
    load_ligand_model,
    load_mrc_volume,
    preprocess_exp_map,
    write_mrc_volume,
)

BOX_SIZE = 64
DEFAULT_LIGAND_WEIGHT = "model_ligand_v0.pt"


def resolve_default_weight_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root.parent / "model_weights" / DEFAULT_LIGAND_WEIGHT,
        repo_root / "model_weights" / DEFAULT_LIGAND_WEIGHT,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No default ligand model weight found. Expected one of: {searched}. "
        "Run scripts/extract_ligand_model_weights.py or pass --model_path."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emready.ligand",
        description="EMReady ligand density inference (experimental map -> ligand maps).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "in_map",
        nargs="?",
        help="Input experimental density map (.mrc/.map). required (unless --input).",
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        help="Output directory for ligand maps. required (unless --output).",
    )
    parser.add_argument("--version", action="version", version=f"EMReady v{emready.__version__}")

    basic_group = parser.add_argument_group("Basic Arguments")
    basic_group.add_argument(
        "--input",
        "-bi",
        "-i",
        dest="in_map_opt",
        help="Input experimental density map (.mrc/.map). default=None.",
    )
    basic_group.add_argument(
        "--output",
        "-bo",
        "-o",
        dest="out_dir_opt",
        help="Output directory for ligand maps. default=None.",
    )
    basic_group.add_argument(
        "--model_path",
        "-bmp",
        "-mp",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    basic_group.add_argument(
        "--stride",
        "-bs",
        "-s",
        type=int,
        default=16,
        help="Sliding-window stride for 64^3 patch inference. valid range: [16, 48]. default=16.",
    )
    basic_group.add_argument(
        "--batch_size",
        "-bb",
        "-b",
        type=int,
        default=8,
        help="Batch size (number of patches per forward pass). default=8.",
    )
    basic_group.add_argument(
        "--gpu_id",
        "-bg",
        "-g",
        type=str,
        default="0",
        help="CUDA visible device id string. Examples: '0', '1', '0,1'. default=0.",
    )
    basic_group.add_argument(
        "--blend_mode",
        "-bbm",
        choices=("uniform", "gaussian"),
        default="gaussian",
        help=argparse.SUPPRESS,
    )
    basic_group.add_argument(
        "--gaussian_sigma_scale",
        "-bgs",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )
    basic_group.add_argument(
        "--exp_percentile",
        type=float,
        default=99.99,
        help=argparse.SUPPRESS,
    )
    basic_group.add_argument(
        "--ligand_class_id",
        type=int,
        default=2,
        help=argparse.SUPPRESS,
    )
    basic_group.add_argument(
        "--class_threshold",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    return parser


def normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    args.in_map = args.in_map_opt or args.in_map
    args.out_dir = args.out_dir_opt or args.out_dir
    if not args.in_map or not args.out_dir:
        parser.error("input map and output directory are required")
    args.in_map = Path(args.in_map)
    args.out_dir = Path(args.out_dir)
    if args.model_path is not None and args.model_path.is_dir():
        parser.error("--model_path/-bmp must be a single weight file, not a directory")
    if args.stride < 16 or args.stride > 48:
        parser.error("--stride/-bs must be in the range [16, 48]")
    if args.stride > BOX_SIZE:
        parser.error(f"--stride/-bs must be <= box size ({BOX_SIZE})")
    if args.gaussian_sigma_scale <= 0:
        parser.error("--gaussian_sigma_scale must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    return args


def build_weight_kernel(blend_mode: str, sigma_scale: float) -> np.ndarray | None:
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


def run(args: argparse.Namespace) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device("cuda")
    print(f"# Running on {torch.cuda.device_count()} GPU(s)")

    weight_file = args.model_path if args.model_path is not None else resolve_default_weight_path()
    if not weight_file.is_file():
        raise FileNotFoundError(f"Model weight file does not exist: {weight_file}")

    model = load_ligand_model(weight_file, device=device)
    torch.cuda.empty_cache()

    exp_map, voxel_size, nxyzstart, origin = load_mrc_volume(args.in_map)
    print(f"# Input map shape [z,y,x]: {exp_map.shape}")
    print(f"# Voxel size [x,y,z]: {voxel_size}")

    exp_norm = preprocess_exp_map(exp_map, percentile=args.exp_percentile)
    weight_kernel = build_weight_kernel(args.blend_mode, args.gaussian_sigma_scale)
    pred_ligand_sim, pred_class, pred_class_prob = infer_ligand_sim_class(
        model=model,
        exp_map_zyx=exp_norm,
        box_size=BOX_SIZE,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
        weight_kernel=weight_kernel,
    )

    ligand_class_id = int(args.ligand_class_id)
    if ligand_class_id < 0 or ligand_class_id >= pred_class_prob.shape[0]:
        raise ValueError(
            f"ligand_class_id={ligand_class_id} out of range [0, {pred_class_prob.shape[0]})"
        )

    pred_ligand = build_ligand_mask(
        pred_ligand_sim,
        pred_class_prob,
        ligand_class_id=ligand_class_id,
        class_threshold=float(args.class_threshold),
    )
    pred_ligand_mask = build_ligand_binary_mask(pred_class, ligand_class_id)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "ligand.mrc": pred_ligand,
        "ligand_mask.mrc": pred_ligand_mask,
    }
    for name, data in outputs.items():
        path = args.out_dir / name
        write_mrc_volume(path, data, voxel_size, nxyzstart, origin)
        print(f"# Saved: {path}")


def main(argv=None) -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args(argv), parser)
    run(args)
    return 0
