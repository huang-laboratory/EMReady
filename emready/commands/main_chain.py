"""EMReady main-chain density inference command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

import emready
from emready.io.mrc import align_origin_to_grid, write_map
from emready.utils.chunks import make_gaussian_weight_kernel
from emready.utils.main_chain_map import (
    infer_main_chain,
    load_main_chain_model,
    normalize_exp_map,
)

BOX_SIZE = 64
DEFAULT_MAIN_CHAIN_WEIGHT = "model_main_chain_v0.pt"


def resolve_default_weight_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root.parent / "model_weights" / DEFAULT_MAIN_CHAIN_WEIGHT,
        repo_root / "model_weights" / DEFAULT_MAIN_CHAIN_WEIGHT,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No default main-chain model weight found. Expected one of: {searched}. "
        "Run scripts/extract_main_chain_model_weights.py or pass --model_path."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emready.main_chain",
        description="EMReady main-chain density inference (experimental map -> main-chain maps).",
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
        help="Output directory for main-chain maps. required (unless --output).",
    )
    parser.add_argument("--version", action="version", version=f"EMReady v{emready.__version__}")

    basic_group = parser.add_argument_group("Basic Arguments")
    basic_group.add_argument("--input", "-bi", "-i", dest="in_map_opt", help="Input experimental density map. default=None.")
    basic_group.add_argument("--output", "-bo", "-o", dest="out_dir_opt", help="Output directory. default=None.")
    basic_group.add_argument("--model_path", "-bmp", "-mp", type=Path, default=None, help=argparse.SUPPRESS)
    basic_group.add_argument("--stride", "-bs", "-s", type=int, default=32, help="Sliding-window stride. valid range: [16, 48]. default=32.")
    basic_group.add_argument("--batch_size", "-bb", "-b", type=int, default=8, help="Batch size. default=8.")
    basic_group.add_argument("--gpu_id", "-bg", "-g", type=str, default="0", help="CUDA visible device id. default=0.")
    basic_group.add_argument("--blend_mode", "-bbm", choices=("uniform", "gaussian"), default="gaussian", help=argparse.SUPPRESS)
    basic_group.add_argument("--gaussian_sigma_scale", "-bgs", type=float, default=0.5, help=argparse.SUPPRESS)
    basic_group.add_argument("--exp_percentile", type=float, default=99.999, help="Percentile clip before z-score. default=99.999.")
    basic_group.add_argument("--contour", type=float, default=1e-6, help="Mask contour threshold. default=1e-6.")
    basic_group.add_argument("--apix", type=float, default=1.0, help="Target voxel size for resampling before prediction. default=1.0.")
    return parser


def normalize_args(args, parser):
    args.in_map = args.in_map_opt or args.in_map
    args.out_dir = args.out_dir_opt or args.out_dir
    if not args.in_map or not args.out_dir:
        parser.error("input map and output directory are required")
    args.in_map = Path(args.in_map)
    args.out_dir = Path(args.out_dir)
    if args.model_path is not None and args.model_path.is_dir():
        parser.error("--model_path must be a single weight file, not a directory")
    if args.stride < 16 or args.stride > 48:
        parser.error("--stride must be in the range [16, 48]")
    if args.gaussian_sigma_scale <= 0:
        parser.error("--gaussian_sigma_scale must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    return args


def build_weight_kernel(blend_mode, sigma_scale):
    if blend_mode == "uniform":
        print("# Patch blending mode: uniform")
        return None
    print(f"# Patch blending mode: gaussian (sigma_scale={sigma_scale})")
    return make_gaussian_weight_kernel(BOX_SIZE, min_weight=1.0, max_weight=3.0, sigma_scale=sigma_scale)


def run(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device("cuda")
    print(f"# Running on {torch.cuda.device_count()} GPU(s)")

    weight_file = args.model_path if args.model_path is not None else resolve_default_weight_path()
    if not weight_file.is_file():
        raise FileNotFoundError(f"Model weight file does not exist: {weight_file}")
    model = load_main_chain_model(weight_file, device=device)
    torch.cuda.empty_cache()

    print("# Loading the input map...")
    map_volume, origin, nxyz, voxel_size, _ = align_origin_to_grid(args.in_map, apix=args.apix)
    nxyzstart = np.round(origin / voxel_size).astype(np.int64)
    print(f"# Map dimensions at {args.apix} A: {nxyz}")

    exp_norm = normalize_exp_map(map_volume, percentile=args.exp_percentile)
    weight_kernel = build_weight_kernel(args.blend_mode, args.gaussian_sigma_scale)
    pred_mc, pred_class = infer_main_chain(
        model=model,
        exp_map_zyx=exp_norm,
        box_size=BOX_SIZE,
        stride=args.stride,
        batch_size=args.batch_size,
        device=device,
        weight_kernel=weight_kernel,
    )

    contour = float(args.contour)
    if contour > 0.0:
        mask = (map_volume > contour).astype(np.float32, copy=False)
        pred_mc = (pred_mc * mask).astype(np.float32, copy=False)
        pred_class = np.where(mask > 0.0, pred_class, 0.0).astype(np.float32, copy=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mc_out = args.out_dir / "main_chain.mrc"
    class_out = args.out_dir / "main_chain_class.mrc"
    write_map(mc_out, pred_mc, voxel_size, nxyzstart=nxyzstart)
    write_map(class_out, pred_class, voxel_size, nxyzstart=nxyzstart)
    print(f"# Saved: {mc_out}")
    print(f"# Saved: {class_out}")


def main(argv=None) -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args(argv), parser)
    run(args)
    return 0
