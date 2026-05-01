# Copyright (C) 2025 Hong Cao, Yueting Zhu et al.

# MIT License

# Copyright (c) 2025 Huang Laboratory

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
import torch.nn as nn

from einops import rearrange
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from emready.models.frn import FilterResponseNorm3d
from emready.vendor.bimamba_ssm import Mamba


class MambaLayer(nn.Module):
    def __init__(self, input_dim, d_state=16, d_conv=4, expand=2, patch_size=4):
        """
        Args:
            input_dim (int): Number of input feature dimensions (channels).
            d_state (int): Hidden state dimension in Mamba. Controls memory capacity.
            d_conv (int): Kernel size for Mamba's convolutional projection. Affects local context modeling.
            expand (int): Expansion ratio for internal projection in Mamba block.
            patch_size (int): Spatial patch size used to flatten 3D volumes.
        """
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(self.input_dim)
        self.mamba = Mamba(
            d_model=self.input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v2",
        )
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.conv_in = nn.Conv3d(
            input_dim * self.patch_size**3, input_dim, 1, 1, 0, bias=True
        )
        self.conv_out = nn.Conv3d(
            input_dim, input_dim * self.patch_size**3, 1, 1, 0, bias=True
        )

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        batch_size, num_channels, D, H, W = x.shape
        x = x.reshape(
            batch_size,
            num_channels,
            D // self.patch_size,
            self.patch_size,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        x = self.conv_in(x)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = x_mamba.reshape(B, C, *img_dims)
        x_mamba = self.conv_out(x_mamba)
        x_mamba = rearrange(
            x_mamba,
            "b (c p1 p2 p3) w1 w2 w3 -> b c (w1 p1) (w2 p2) (w3 p3)",
            w1=D // self.patch_size,
            w2=H // self.patch_size,
            w3=W // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            p3=self.patch_size,
        )
        return x_mamba


class BiMambaBlock(nn.Module):

    def __init__(self, spatial_dims, in_channels, norm, act, patch_size):
        """
        Bidirectional Mamba Block (BiMambaBlock), designed to capture spatiotemporal features
        using two sequential Mamba layers with normalization and activation in between.

        Args:
            spatial_dims (int): Number of spatial dimensions (1, 2, or 3).
            in_channels (int): Number of input channels/features.
            norm (tuple | str): Normalization type and its arguments. Passed to `get_norm_layer`.
            act (tuple | str): Activation function type and its arguments. Passed to `get_act_layer`.
            patch_size (int): Patch size used in the MambaLayer for spatial flattening.
        """
        super().__init__()
        self.norm1 = get_norm_layer(
            name=norm, spatial_dims=spatial_dims, channels=in_channels
        )
        self.norm2 = get_norm_layer(
            name=norm, spatial_dims=spatial_dims, channels=in_channels
        )
        self.act = get_act_layer(act)
        self.mamba1 = MambaLayer(input_dim=in_channels, patch_size=patch_size)
        self.mamba2 = MambaLayer(input_dim=in_channels, patch_size=patch_size)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.mamba1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.mamba2(x)
        x += identity
        return x


class BiMCBlock(nn.Module):
    def __init__(self, conv_dim, mamba_dim, patch_size):
        """
        A dual-branch block combining convolutional and BiMamba branches.

        Args:
            conv_dim (int): Number of channels for the convolutional branch.
            mamba_dim (int): Number of channels for the Mamba branch.
            patch_size (int): Patch size passed to the BiMambaBlock for spatial processing.
        """
        super(BiMCBlock, self).__init__()
        self.conv_dim = conv_dim
        self.mamba_dim = mamba_dim
        self.mamba_block = BiMambaBlock(
            3,
            mamba_dim,
            norm=("GROUP", {"num_groups": 8}),
            act=("RELU", {"inplace": True}),
            patch_size=patch_size,
        )
        self.conv1_1 = nn.Conv3d(
            self.conv_dim + self.mamba_dim,
            self.conv_dim + self.mamba_dim,
            1,
            1,
            0,
            bias=True,
        )
        self.conv1_2 = nn.Conv3d(
            self.conv_dim + self.mamba_dim,
            self.conv_dim + self.mamba_dim,
            1,
            1,
            0,
            bias=True,
        )
        self.conv_block = nn.Sequential(
            nn.Conv3d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            FilterResponseNorm3d(self.conv_dim),
            nn.Conv3d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            FilterResponseNorm3d(self.conv_dim),
        )

    def forward(self, x):
        conv_x, mamba_x = torch.split(
            self.conv1_1(x), (self.conv_dim, self.mamba_dim), dim=1
        )
        conv_x = self.conv_block(conv_x) + conv_x
        mamba_x = self.mamba_block(mamba_x)
        res = self.conv1_2(torch.cat((conv_x, mamba_x), dim=1))
        x = x + res
        return x


class BiMCUnet(nn.Module):

    def __init__(
        self, in_nc=1, config=[2, 2, 2, 2, 2, 2, 2], dim=32, out_nc=1, patch_size=4
    ):
        """
        Args:
            in_nc (int): Number of input channels.
            config (list of int): Number of BiMC blocks at each stage. Expected length is 7:
                [down1, down2, down3, body, up3, up2, up1].
            dim (int): Base channel dimension.
            out_nc (int): Number of output channels.
            patch_size (int): Patch size used in each BiMCBlock.
        """
        super(BiMCUnet, self).__init__()
        self.config = config
        self.dim = dim
        self.m_head = [nn.Conv3d(in_nc, dim, 3, 1, 1, bias=False)]
        self.m_down1 = [
            BiMCBlock(dim // 2, dim // 2, patch_size) for i in range(config[0])
        ] + [nn.Conv3d(dim, 2 * dim, 2, 2, 0, bias=False)]
        self.m_down2 = [BiMCBlock(dim, dim, patch_size) for i in range(config[1])] + [
            nn.Conv3d(2 * dim, 4 * dim, 2, 2, 0, bias=False)
        ]
        self.m_down3 = [
            BiMCBlock(2 * dim, 2 * dim, patch_size) for i in range(config[2])
        ] + [nn.Conv3d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]
        self.m_body = [
            BiMCBlock(4 * dim, 4 * dim, patch_size) for i in range(config[3])
        ]
        self.m_up3 = [
            nn.ConvTranspose3d(8 * dim, 4 * dim, 2, 2, 0, bias=False),
        ] + [BiMCBlock(2 * dim, 2 * dim, patch_size) for i in range(config[4])]
        self.m_up2 = [
            nn.ConvTranspose3d(4 * dim, 2 * dim, 2, 2, 0, bias=False),
        ] + [BiMCBlock(dim, dim, patch_size) for i in range(config[5])]
        self.m_up1 = [
            nn.ConvTranspose3d(2 * dim, dim, 2, 2, 0, bias=False),
        ] + [BiMCBlock(dim // 2, dim // 2, patch_size) for i in range(config[6])]
        self.m_tail = [nn.Conv3d(dim, out_nc, 3, 1, 1, bias=False)]
        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x
