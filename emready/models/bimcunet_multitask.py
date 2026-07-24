"""BiMCUnet multi-task head for ligand density inference."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from emready.vendor.bimamba_ssm import Mamba


def _group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class MambaLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        patch_size: int = 4,
    ) -> None:
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
        self.conv_in = nn.Conv3d(input_dim * self.patch_size**3, input_dim, 1, 1, 0, bias=True)
        self.conv_out = nn.Conv3d(input_dim, input_dim * self.patch_size**3, 1, 1, 0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        batch_size, num_channels, depth, height, width = x.shape
        patch = self.patch_size
        x = x.reshape(
            batch_size,
            num_channels,
            depth // patch,
            patch,
            height // patch,
            patch,
            width // patch,
            patch,
        )
        x = x.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        x = self.conv_in(x)
        batch, channels = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(batch, channels, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = x_mamba.reshape(batch, channels, *img_dims)
        x_mamba = self.conv_out(x_mamba)
        return rearrange(
            x_mamba,
            "b (c p1 p2 p3) w1 w2 w3 -> b c (w1 p1) (w2 p2) (w3 p3)",
            w1=depth // patch,
            w2=height // patch,
            w3=width // patch,
            p1=patch,
            p2=patch,
            p3=patch,
        )


class BiMambaBlock(nn.Module):
    def __init__(self, in_channels: int, patch_size: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.norm2 = _group_norm(in_channels)
        self.act = nn.SiLU(inplace=True)
        self.mamba1 = MambaLayer(input_dim=in_channels, patch_size=patch_size)
        self.mamba2 = MambaLayer(input_dim=in_channels, patch_size=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.mamba1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.mamba2(x)
        return x + identity


class ConvGNBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            _group_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            _group_norm(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BiMCBlock(nn.Module):
    def __init__(self, conv_dim: int, mamba_dim: int, patch_size: int) -> None:
        super().__init__()
        self.conv_dim = conv_dim
        self.mamba_dim = mamba_dim
        self.mamba_block = BiMambaBlock(mamba_dim, patch_size)
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
        self.conv_block = ConvGNBlock(self.conv_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_x, mamba_x = torch.split(self.conv1_1(x), (self.conv_dim, self.mamba_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        mamba_x = self.mamba_block(mamba_x)
        res = self.conv1_2(torch.cat((conv_x, mamba_x), dim=1))
        return x + res


class BiMCUnetLigand(nn.Module):
    def __init__(
        self,
        in_nc: int = 1,
        config: list[int] | tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2),
        dim: int = 32,
        out_nc: int = 4,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.config = list(config)
        self.dim = dim

        self.m_head = nn.Sequential(nn.Conv3d(in_nc, dim, 3, 1, 1, bias=False))
        self.m_down1 = nn.Sequential(
            *(
                [BiMCBlock(dim // 2, dim // 2, patch_size) for _ in range(self.config[0])]
                + [nn.Conv3d(dim, 2 * dim, 2, 2, 0, bias=False)]
            )
        )
        self.m_down2 = nn.Sequential(
            *(
                [BiMCBlock(dim, dim, patch_size) for _ in range(self.config[1])]
                + [nn.Conv3d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]
            )
        )
        self.m_down3 = nn.Sequential(
            *(
                [BiMCBlock(2 * dim, 2 * dim, patch_size) for _ in range(self.config[2])]
                + [nn.Conv3d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]
            )
        )
        self.m_body = nn.Sequential(
            *[BiMCBlock(4 * dim, 4 * dim, patch_size) for _ in range(self.config[3])]
        )
        self.m_up3 = nn.Sequential(
            *(
                [nn.ConvTranspose3d(8 * dim, 4 * dim, 2, 2, 0, bias=False)]
                + [BiMCBlock(2 * dim, 2 * dim, patch_size) for _ in range(self.config[4])]
            )
        )
        self.m_up2 = nn.Sequential(
            *(
                [nn.ConvTranspose3d(4 * dim, 2 * dim, 2, 2, 0, bias=False)]
                + [BiMCBlock(dim, dim, patch_size) for _ in range(self.config[5])]
            )
        )
        self.m_up1 = nn.Sequential(
            *(
                [nn.ConvTranspose3d(2 * dim, dim, 2, 2, 0, bias=False)]
                + [BiMCBlock(dim // 2, dim // 2, patch_size) for _ in range(self.config[6])]
            )
        )
        self.m_tail = nn.Sequential(nn.Conv3d(dim, out_nc, 3, 1, 1, bias=False))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)


class BiMCUnetMultiTask(nn.Module):
    """Ligand similarity + 3-class segmentation head (s1de_v3 j3)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        base_dim: int = 32,
        patch_size: int = 4,
        block_config: list[int] | tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2),
    ) -> None:
        super().__init__()
        self.network = BiMCUnetLigand(
            in_nc=in_channels,
            config=block_config,
            dim=base_dim,
            out_nc=out_channels,
            patch_size=patch_size,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y = self.network(x)
        out_channels = y.shape[1]
        if out_channels == 1:
            return {"sim": y, "all": y}
        if out_channels == 3:
            return {"class_logits": y, "all": y}
        return {
            "class_logits": y[:, 0:3, ...],
            "sim": y[:, 3:4, ...],
            "all": y,
        }
