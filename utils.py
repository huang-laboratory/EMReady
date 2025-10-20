# Copyright (C) 2024 Hong Cao, Jiahua He, Tao Li, Sheng-You Huang and Huazhong University of Science and Technology

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import mrcfile
import numpy as np
from math import ceil

from interp3d import Interp3D

interp3d = Interp3D()


def split_map_into_overlapped_chunks(map, box_size, stride, dtype=np.float32, padding=0.0):
    assert stride <= box_size
    map_shape = np.shape(map)
    padded_map = np.full((map_shape[0] + 2 * box_size, map_shape[1] + 2 * box_size, map_shape[2] + 2 * box_size), padding, dtype=dtype)
    padded_map[box_size : box_size + map_shape[0], box_size : box_size + map_shape[1], box_size : box_size + map_shape[2]] = map
    chunk_list = list()
    start_point = box_size - stride
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    while (cur_z + stride < map_shape[2] + box_size):
        next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        cur_x += stride
        if (cur_x + stride >= map_shape[0] + box_size):
            cur_y += stride
            cur_x = start_point # Reset X
            if (cur_y + stride  >= map_shape[1] + box_size):
                cur_z += stride
                cur_y = start_point # Reset Y
                cur_x = start_point # Reset X
        chunk_list.append(next_chunk)
    n_chunks = len(chunk_list)
    ncx, ncy, ncz = [ceil(map_shape[i] / stride) for i in range(3)]
    assert(n_chunks == ncx * ncy * ncz)
    chunks = np.asarray(chunk_list, dtype=dtype)
    return chunks, ncx, ncy, ncz


# get a map from all of the overlapped chunks
def get_map_from_overlapped_chunks(chunks, ncx, ncy, ncz, box_size, stride, nxyz, dtype=np.float32):
    map = np.zeros(((ncx - 1) * stride + box_size, \
                    (ncy - 1) * stride + box_size, \
                    (ncz - 1) * stride + box_size), dtype=dtype)
    denominator = np.zeros(((ncx - 1) * stride + box_size, \
                            (ncy - 1) * stride + box_size, \
                            (ncz - 1) * stride + box_size), dtype=dtype) # should clip to 1
    i = 0
    for z_steps in range(ncz):
        for y_steps in range(ncy):
            for x_steps in range(ncx):
                map[x_steps * stride : x_steps * stride + box_size,
                    y_steps * stride : y_steps * stride + box_size,
                    z_steps * stride : z_steps * stride + box_size] += chunks[i]
                denominator[x_steps * stride : x_steps * stride + box_size,
                            y_steps * stride : y_steps * stride + box_size,
                            z_steps * stride : z_steps * stride + box_size] += 1
                    
                i += 1
    return (map / denominator.clip(min=1))[stride : nxyz[2] + stride, stride : nxyz[1] + stride, stride : nxyz[0] + stride]


def pad_map(map, box_size, dtype=np.float32, padding=0.0):
    map_shape = np.shape(map)
    padded_map = np.full((map_shape[0] + 2 * box_size, map_shape[1] + 2 * box_size, map_shape[2] + 2 * box_size), padding, dtype=dtype)
    padded_map[box_size : box_size + map_shape[0], box_size : box_size + map_shape[1], box_size : box_size + map_shape[2]] = map
    return padded_map


# generator version
def chunk_generator(padded_map, maximum, box_size, stride):
    assert stride <= box_size
    padded_map_shape = np.shape(padded_map)
    start_point = box_size - stride
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    while (cur_z + stride < padded_map_shape[2] - box_size):
        next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        cur_x0, cur_y0, cur_z0 = cur_x, cur_y, cur_z
        cur_x += stride
        if (cur_x + stride >= padded_map_shape[0] - box_size):
            cur_y += stride
            cur_x = start_point # Reset X
            if (cur_y + stride  >= padded_map_shape[1] - box_size):
                cur_z += stride
                cur_y = start_point # Reset Y
                cur_x = start_point # Reset X

        if next_chunk.max() <= 0.0:
            continue
        else:
            yield cur_x0, cur_y0, cur_z0, next_chunk.clip(min=0.0, max=maximum) / maximum * 100.0


# get a batch of chunks from generator
def get_batch_from_generator(generator, batch_size, dtype=np.float32):
    positions = list()
    batch = list()
    for _ in range(batch_size):
        try:
            output = next(generator)
            positions.append(output[:3])
            batch.append(output[3])
        except StopIteration:
            break
    return positions, np.asarray(batch, dtype=dtype)


# map the batch of chunks to the map
def map_batch_to_map(pred_map, denominator, positions, batch, box_size):
    for position, chunk in zip(positions, batch):
        pred_map[position[0]:position[0] + box_size, position[1]:position[1] + box_size, position[2]:position[2] + box_size] += chunk
        denominator[position[0]:position[0] + box_size, position[1]:position[1] + box_size, position[2]:position[2] + box_size] += 1
    return pred_map, denominator


def parse_map(map_file, ignorestart, apix=None, origin_shift=None):

    ''' parse mrc '''
    mrc = mrcfile.open(map_file, mode='r')

    map = np.asarray(mrc.data.copy(), dtype=np.float32)
    voxel_size = np.asarray([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    ncrsstart = np.asarray([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)
    origin = np.asarray([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=np.float32)
    ncrs = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
    angle = np.asarray([mrc.header.cellb.alpha, mrc.header.cellb.beta, mrc.header.cellb.gamma], dtype=np.float32)

    ''' check orthogonal '''
    try:
        assert(angle[0] == angle[1] == angle[2] == 90.0)
    except AssertionError:
        print("# Input grid is not orthogonal. EXIT.")
        mrc.close()
        exit()

    ''' reorder axes '''
    mapcrs = np.subtract([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], 1)
    sort = np.asarray([0, 1, 2], dtype=np.int64)
    for i in range(3):
        sort[mapcrs[i]] = i
    nxyzstart = np.asarray([ncrsstart[i] for i in sort])
    nxyz = np.asarray([ncrs[i] for i in sort])
    nxyz_old = nxyz

    map = np.transpose(map, axes=2-sort[::-1])
    mrc.close()

    ''' shift origin according to n*start '''
    if not ignorestart:
        origin += np.multiply(nxyzstart, voxel_size)

    ''' interpolate grid interval '''
    if apix is not None:
        try:
            assert(voxel_size[0] == voxel_size[1] == voxel_size[2] == apix and origin_shift is None)
        except AssertionError:
            interp3d.del_mapout()
            target_voxel_size = np.asarray([apix, apix, apix], dtype=np.float32)
            print("# Rescale voxel size from {} to {}".format(voxel_size, target_voxel_size))
            if origin_shift is not None:
                interp3d.cubic(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, origin_shift[2], origin_shift[1], origin_shift[0], nxyz[2], nxyz[1], nxyz[0])
                origin += origin_shift
            else:
                interp3d.cubic(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, 0.0, 0.0, 0.0, nxyz[2], nxyz[1], nxyz[0])
                
            map = interp3d.mapout
            nxyz = np.asarray([interp3d.pextx, interp3d.pexty, interp3d.pextz], dtype=np.int64)
            voxel_size = target_voxel_size

    assert(np.all(nxyz == np.asarray([map.shape[2], map.shape[1], map.shape[0]], dtype=np.int64)))

    return map, origin, nxyz, voxel_size, nxyz_old


def parse_and_shift_mrc(mrc_path, ignorestart=False, apix=None):
    map, origin, nxyz, voxel_size, nxyz_old = parse_map(mrc_path, ignorestart=ignorestart, apix=apix)
    try:
        assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)
    except AssertionError:
        origin_shift = voxel_size * (np.round(origin / voxel_size) - origin / voxel_size)
        map, origin, nxyz, voxel_size, nxyz_old = parse_map(mrc_path, ignorestart=ignorestart, apix=apix, origin_shift=origin_shift)
    assert np.all(np.abs(np.round(origin / voxel_size) - origin / voxel_size) < 1e-4)
    nxyzstart = np.round(origin / voxel_size).astype(np.int64)
    return map, voxel_size, nxyz, nxyzstart


def inverse_map(map_pred, nxyz, origin, voxel_size, old_voxel_size, origin_shift):
    interp3d.del_mapout()
    interp3d.inverse_cubic(map_pred, voxel_size[2], voxel_size[1], voxel_size[0], old_voxel_size[2], old_voxel_size[1], old_voxel_size[0], origin_shift[2], origin_shift[1], origin_shift[0], nxyz[2], nxyz[1], nxyz[0])
    origin += origin_shift
    nxyz = np.asarray([interp3d.pextx, interp3d.pexty, interp3d.pextz], dtype=np.int64)
    map_pred = interp3d.mapout
    voxel_size = old_voxel_size
    return map_pred, origin, nxyz, voxel_size


def write_map(file_name, map, voxel_size, origin=(0.0, 0.0, 0.0), nxyzstart=(0, 0, 0)):
    mrc = mrcfile.new(file_name, overwrite=True)
    mrc.set_data(map)
    (mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart) = nxyzstart
    (mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z) = origin
    mrc.voxel_size = [voxel_size[i] for i in range(3)]
    mrc.close()
