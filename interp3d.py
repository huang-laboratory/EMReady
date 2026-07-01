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

import numpy as np
from numba import jit, prange


@jit(nopython=True)
def _get_w(x, w):
    a = -0.5
    intx = int(np.floor(x))
    d1 = 1.0 + (x - intx)
    d2 = d1 - 1.0
    d3 = 1.0 - d2
    d4 = d3 + 1.0
    w[0] = a * np.abs(d1**3) - 5 * a * d1**2 + 8 * a * np.abs(d1) - 4 * a
    w[1] = (a + 2) * np.abs(d2**3) - (a + 3) * d2**2 + 1
    w[2] = (a + 2) * np.abs(d3**3) - (a + 3) * d3**2 + 1
    w[3] = a * np.abs(d4**3) - 5 * a * d4**2 + 8 * a * np.abs(d4) - 4 * a


class Interp3D:
    def __init__(self):
        self.mapout = None
        self.pextx = None
        self.pexty = None
        self.pextz = None

    def cubic(self, mapin, zpix, ypix, xpix, apix, shiftz, shifty, shiftx, nz, ny, nx):
        pextx = int(np.floor(xpix * (nx - 1) / apix)) + 1
        pexty = int(np.floor(ypix * (ny - 1) / apix)) + 1
        pextz = int(np.floor(zpix * (nz - 1) / apix)) + 1
        self.mapout = np.zeros((pextz, pexty, pextx), dtype=np.float32)
        self.pextx = pextx
        self.pexty = pexty
        self.pextz = pextz
        self.mapout = self._cubic_interp(mapin, zpix, ypix, xpix, apix, shiftz, shifty, shiftx, nz, ny, nx, pextz, pexty, pextx, self.mapout)
        return self.mapout

    @staticmethod
    @jit(parallel=True, nopython=True)
    def _cubic_interp(mapin, zpix, ypix, xpix, apix, shiftz, shifty, shiftx, nz, ny, nx, pextz, pexty, pextx, mapout):
        for indz in prange(pextz):  # 0 ~ pextz-1
            for indy in prange(pexty):  # 0 ~ pexty-1
                for indx in prange(pextx):  # 0 ~ pextx-1
                    gx = (indx * apix + shiftx) / xpix
                    gy = (indy * apix + shifty) / ypix
                    gz = (indz * apix + shiftz) / zpix
                    intx = int(np.floor(gx))
                    inty = int(np.floor(gy))
                    intz = int(np.floor(gz))
                    if intz >= 0 and intz + 1 < nz and inty >= 0 and inty + 1 < ny and intx >= 0 and intx + 1 < nx:
                        wz = np.zeros(4, dtype=np.float32)
                        wy = np.zeros(4, dtype=np.float32)
                        wx = np.zeros(4, dtype=np.float32)
                        _get_w(gz, wz)
                        _get_w(gy, wy)
                        _get_w(gx, wx)
                        for i in range(4):
                            for j in range(4):
                                for k in range(4):
                                    if (intz + i - 1 >= 0 and intz + i - 1 < nz and
                                        inty + j - 1 >= 0 and inty + j - 1 < ny and
                                        intx + k - 1 >= 0 and intx + k - 1 < nx):
                                        mapout[indz, indy, indx] += wz[i] * wy[j] * wx[k] * mapin[intz + i - 1, inty + j - 1, intx + k - 1]
        return mapout

    def inverse_cubic(self, mapin, zpix, ypix, xpix, zpix_o, ypix_o, xpix_o, shiftz, shifty, shiftx, nz, ny, nx):
        pextx = int(np.ceil(xpix * (nx - 1) / xpix_o)) + 1
        pexty = int(np.ceil(ypix * (ny - 1) / ypix_o)) + 1
        pextz = int(np.ceil(zpix * (nz - 1) / zpix_o)) + 1
        self.mapout = np.zeros((pextz, pexty, pextx), dtype=np.float32)
        self.pextx = pextx
        self.pexty = pexty
        self.pextz = pextz
        self.mapout = self._inverse_cubic_interp(mapin, zpix, ypix, xpix, zpix_o, ypix_o, xpix_o, shiftz, shifty, shiftx, nz, ny, nx, pextz, pexty, pextx, self.mapout)
        return self.mapout

    @staticmethod
    @jit(parallel=True, nopython=True)
    def _inverse_cubic_interp(mapin, zpix, ypix, xpix, zpix_o, ypix_o, xpix_o, shiftz, shifty, shiftx, nz, ny, nx, pextz, pexty, pextx, mapout):
        for indz in prange(pextz):  # 0 ~ pextz-1
            for indy in prange(pexty):  # 0 ~ pexty-1
                for indx in prange(pextx):  # 0 ~ pextx-1
                    gx = (indx * xpix_o + shiftx) / xpix
                    gy = (indy * ypix_o + shifty) / ypix
                    gz = (indz * zpix_o + shiftz) / zpix
                    intx = int(np.floor(gx))
                    inty = int(np.floor(gy))
                    intz = int(np.floor(gz))
                    if intz >= 0 and intz + 1 < nz and inty >= 0 and inty + 1 < ny and intx >= 0 and intx + 1 < nx:
                        wz = np.zeros(4, dtype=np.float32)
                        wy = np.zeros(4, dtype=np.float32)
                        wx = np.zeros(4, dtype=np.float32)
                        _get_w(gz, wz)
                        _get_w(gy, wy)
                        _get_w(gx, wx)
                        for i in range(4):
                            for j in range(4):
                                for k in range(4):
                                    if (intz + i - 1 >= 0 and intz + i - 1 < nz and
                                        inty + j - 1 >= 0 and inty + j - 1 < ny and
                                        intx + k - 1 >= 0 and intx + k - 1 < nx):
                                        mapout[indz, indy, indx] += wz[i] * wy[j] * wx[k] * mapin[intz + i - 1, inty + j - 1, intx + k - 1]
        return mapout

    def del_mapout(self):
        self.mapout = None
