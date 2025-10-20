# Copyright (C) 2024  Hong Cao, Jiahua He, Tao Li, Sheng-You Huang and Huazhong University of Science and Technology

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import torch
import argparse
import warnings
import numpy as np
from torch import nn
from math import ceil
from Bio.PDB import PDBParser
from Bio.PDB import MMCIFParser
from Bio import BiopythonWarning
from torch import FloatTensor as FT
from torch.autograd import Variable as V
from utils import parse_map, pad_map, chunk_generator, get_batch_from_generator, map_batch_to_map, write_map, inverse_map

from scunet import SCUNet

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', BiopythonWarning)

def get_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--in_map", "-i", type=str, required=True)
    parser.add_argument("--out_map", "-o", type=str, required=True)
    parser.add_argument("--model_dir", "-md", type=str, required=True)
    parser.add_argument("--stride", "-s", type=int, default=12)
    parser.add_argument("--batch_size", "-b", type=int, default=10)
    parser.add_argument("--gpu_id", "-g", type=str, default="0")
    parser.add_argument("--mask_map", "-m", type=str, default=None)
    parser.add_argument("--mask_contour", "-c", type=float, default=0.0)
    parser.add_argument("--mask_str", "-p", type=str, default=None)
    parser.add_argument("--mask_str_radius", "-r", type=float, default=4.0)
    parser.add_argument("--mask_out", "-mo", type=str, default=None)
    parser.add_argument("--use_cpu", action='store_true', default=False)
    parser.add_argument("--inverse_mask", action='store_true', default=False)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    in_map = args.in_map
    out_map = args.out_map
    mask_map = args.mask_map
    mask_contour = args.mask_contour
    mask_str = args.mask_str
    mask_str_radius = args.mask_str_radius
    mask_out = args.mask_out
    inverse_mask = args.inverse_mask
    gpu_id = args.gpu_id
    batch_size = args.batch_size
    stride = args.stride
    use_cpu = args.use_cpu
    model_dir = args.model_dir

    BOX_SIZE = 48
    PERCENTILE = 99.999

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"# Running on {n_gpus} GPU(s)")
        else:
            raise RuntimeError("CUDA not available")
    else:
        n_gpus = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("# Running on CPU")

    if not (48 >= stride >= 6):
        raise ValueError("`--stride` (`-s`) must be in the range of [6, 48]")

    if mask_map is not None and mask_str is not None:
        raise ValueError("`--mask_map` (`-m`) and `--mask_str` (`-p`) cannot be provided at the same time")

    _, _, _, voxel_size, _ = parse_map(in_map, ignorestart=False)
    print(f"# Voxel size of the input map: {voxel_size}")

    if voxel_size.min() >= 1.0:
        print("# Using the EMReady model of 1.0 Angstrom grid size")
        apix = 1.0
        model_state_dict_file = f"{model_dir}/model_grid_size_1.0.pth"
    else:
        print("# Using the EMReady model of 0.5 Angstrom grid size")
        apix = 0.5
        model_state_dict_file = f"{model_dir}/model_grid_size_0.5.pth"

    if not use_cpu:
        model_state_dict = torch.load(model_state_dict_file)
    else:
        model_state_dict = torch.load(model_state_dict_file, map_location=torch.device('cpu'))

    model = SCUNet(
        in_nc=1,
        config=[2,2,2,2,2,2,2],
        dim=32,
        drop_path_rate=0.0,
        input_resolution=48,
        head_dim=16,
        window_size=3,
    )

    model.load_state_dict(model_state_dict)

    if not use_cpu:
        torch.cuda.empty_cache()
        model = model.cuda()
        if n_gpus > 1:
            model = nn.DataParallel(model)

    model.eval()

    print("# Loading the input map...")

    map, origin, nxyz, voxel_size, nxyz_origin = parse_map(in_map, ignorestart=False, apix=apix)

    print(f"# Original map dimensions: {nxyz_origin}")

    try:
        assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)

    except AssertionError:
        origin_shift =  ( np.round(origin / voxel_size) - origin / voxel_size ) * voxel_size
        map, origin, nxyz, voxel_size, _ = parse_map(in_map, ignorestart=False, apix=apix, origin_shift=origin_shift)
        assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)

    nxyzstart = np.round(origin / voxel_size).astype(np.int64)

    print(f"# Map dimensions at {apix} Angstrom grid size: {nxyz}")

    map_volume = map.copy()

    del map

    _, _, _, old_voxel_size, _ = parse_map(in_map, ignorestart=False, apix=None)

    if mask_map is not None:
        map_mask = map_volume.copy()
        del map_volume

        print("# Loading the mask map...")

        mask, origin_mask, nxyz_mask, voxel_size_mask, _ = parse_map(mask_map, ignorestart=False, apix=apix)

        try:
            assert np.all(np.abs(np.round(origin_mask / voxel_size_mask) - origin_mask / voxel_size_mask) < 1e-4)

        except AssertionError:
            origin_shift_mask =  ( np.round(origin_mask / voxel_size_mask) - origin_mask / voxel_size_mask ) * voxel_size_mask
            mask, origin_mask, nxyz_mask, voxel_size_mask, _ = parse_map(mask_map, ignorestart=False, apix=apix, origin_shift=origin_shift_mask)
            assert np.all(np.abs(np.round(origin_mask / voxel_size_mask) - origin_mask / voxel_size_mask) < 1e-4)

        nxyzstart_mask = np.round(origin_mask / voxel_size_mask).astype(np.int64)

        print(f"# Mask map dimensions: {nxyz_mask}")

        assert np.all(nxyz_mask <= nxyz)

        try:
            assert np.all(nxyz_mask == nxyz)

        except AssertionError:
            pad_mask = np.zeros(nxyz[::-1]).astype(np.float32)
            nxyz_shift = nxyzstart_mask - nxyzstart
            pad_mask[nxyz_shift[2]:nxyz_shift[2]+nxyz_mask[2], nxyz_shift[1]:nxyz_shift[1]+nxyz_mask[1], nxyz_shift[0]:nxyz_shift[0]+nxyz_mask[0]] = mask
            mask = pad_mask
            origin_mask = origin
            nxyz_mask = nxyz
            nxyzstart_mask = nxyzstart

        if inverse_mask:
            map_volume = np.where(mask < mask_contour, map_mask, 0).astype(np.float32)
        else:
            map_volume = np.where(mask >= mask_contour, map_mask, 0).astype(np.float32)

        if mask_out is not None:
            if inverse_mask:
                mask_o = np.where(mask < mask_contour, 1, 0).astype(np.float32)
            else:
                mask_o = np.where(mask >= mask_contour, 1, 0).astype(np.float32)

            print(f"# Saving the binary mask map to {mask_out}")
            write_map(mask_out, mask_o, voxel_size_mask, nxyzstart=nxyzstart_mask)

        del map_mask, mask

    if mask_str is not None:
        map_mask = map_volume.copy()
        del map_volume

        if mask_str.split(".")[-1][-3:] == "pdb" or mask_str.split(".")[-1][-4:] == "pdb1":
            parser = PDBParser()
        elif mask_str.split(".")[-1][-3:] == "cif":
            parser = MMCIFParser()
        else:
            raise RuntimeError("Unknown type for structure file:", mask_str[-3:])
        structures = parser.get_structure("str", mask_str)
        coords = []

        structure = structures[0]
        for atom in structure.get_atoms():
            if atom.element == 'H':
                continue
            coords.append(atom.get_coord())
        atoms = np.asarray(coords, dtype=np.float32)
        del coords

        print(f"# Generating the mask map from the structure file {mask_str}...")
        map_volume = np.zeros(nxyz[::-1], dtype=np.float32)
        mask = np.zeros(nxyz[::-1], dtype=np.int16)
        for atom in atoms:
            atom_shifted = atom - origin
            lower = np.floor((atom_shifted - mask_str_radius) / voxel_size).astype(np.int32)
            upper = np.ceil ((atom_shifted + mask_str_radius) / voxel_size).astype(np.int32)
            for x in range(lower[0], upper[0] + 1):
                for y in range(lower[1], upper[1] + 1):
                    for z in range(lower[2], upper[2] + 1):
                        if 0 <= x < nxyz[0] and 0 <= y < nxyz[1] and 0 <= z < nxyz[2]:
                            if mask[z, y, x] == 0:
                                vector = np.array([x, y, z], dtype=np.float32) * voxel_size - atom_shifted
                                dist = np.sqrt(vector@vector)
                                if dist < mask_str_radius:
                                    mask[z, y, x] = 1

        if inverse_mask:
            mask = 1 - mask
        map_volume = map_mask * mask.astype(np.float32)

        if mask_out is not None:
            print(f"# Saving the binary mask map to {mask_out}")
            write_map(mask_out, mask.astype(np.float32), voxel_size, nxyzstart=nxyzstart)

        del map_mask, mask

    map = map_volume.copy()
    del map_volume

    padded_map = pad_map(map, BOX_SIZE, dtype=np.float32, padding=0.0)
    maximum = np.percentile(map[map > 0], PERCENTILE)
    del map

    map_pred = np.zeros_like(padded_map, dtype=np.float32)
    denominator = np.zeros_like(padded_map, dtype=np.float32)

    print("# Start processing...")

    generator = chunk_generator(padded_map, maximum, BOX_SIZE, stride)
    ncx, ncy, ncz = [ceil(nxyz[2-i] / stride) for i in range(3)]
    total_steps = float(ncx * ncy * ncz)
    acc_steps, acc_steps_x, l_bar = 0.0, 0, 0

    with torch.inference_mode():
        while True:
            positions, chunks = get_batch_from_generator(generator, batch_size, dtype=np.float32)

            if len(positions) == 0:
                break

            acc_steps += len(chunks)
            acc_steps_x = int((acc_steps / total_steps) * 100.0) // 5
            if acc_steps_x >= l_bar:
                l_bar = acc_steps_x
                bar = f"|{'#' * (2*l_bar)}{'-' * ((20-l_bar)*2)}| {int(l_bar*5)}%"
                print(f"\r{bar}", flush=True)

            X = V(FT(chunks), requires_grad=False).view(-1, 1, BOX_SIZE, BOX_SIZE, BOX_SIZE)
            if not use_cpu:
                X = X.cuda()

            y_pred = model(X).view(-1, BOX_SIZE, BOX_SIZE, BOX_SIZE)
            y_pred = y_pred.cpu().detach().numpy()

            map_pred, denominator = map_batch_to_map(map_pred, denominator, positions, y_pred, BOX_SIZE)

    map_pred = (map_pred/denominator.clip(min=1))[BOX_SIZE:BOX_SIZE+nxyz[2], BOX_SIZE:BOX_SIZE+nxyz[1], BOX_SIZE:BOX_SIZE+nxyz[0]]

    if acc_steps < total_steps:
        bar = f"|{'#' * 40}| 100%"
        print(f"\r{bar}", flush=True)

    print(f"# Interpolating the voxel size from {voxel_size} back to {old_voxel_size}")
    origin = nxyzstart * voxel_size
    origin_shift = [0.0, 0.0, 0.0]
    try:
        assert np.all(np.abs(np.round(origin / old_voxel_size) - origin / old_voxel_size) < 1e-3)
    except AssertionError:
        origin_shift = ( np.round(origin / old_voxel_size) - origin / old_voxel_size ) * old_voxel_size
    map_pred, origin, nxyz, voxel_size = inverse_map(map_pred, nxyz, origin, voxel_size, old_voxel_size, origin_shift)
    assert np.all(np.abs(np.round(origin / old_voxel_size) - origin / old_voxel_size) < 1e-3)
    nxyzstart = np.round(origin / voxel_size).astype(np.int64)

    print(f"# Saving the processed map that has been interpolated back to the original grid size to {out_map}")
    write_map(out_map, map_pred, voxel_size, nxyzstart=nxyzstart)



if __name__ == "__main__":
    main()
