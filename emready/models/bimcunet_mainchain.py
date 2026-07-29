"""BiMCUnet multi-task head for main-chain density prediction (emalign stage1)."""

from __future__ import annotations

import torch
import torch.nn as nn

from emready.models.bimcunet import BiMCUnet


class BiMCUnetMainTask(nn.Module):
    """Joint main-chain density + 3-class segmentation head (out_channels=4).

    Output channel layout: [0:3] = class_logits, [3:4] = main-chain density.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        base_dim: int = 32,
        patch_size: int = 4,
        block_config: list[int] | tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2),
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.network = BiMCUnet(
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
            return {"mc": y, "all": y}
        if out_channels == 3:
            return {"class_logits": y, "all": y}
        return {
            "class_logits": y[:, 0:3, ...],
            "mc": y[:, 3:4, ...],
            "all": y,
        }
