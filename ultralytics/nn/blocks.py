# models/sdi_unetv2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SDI_UNetV2(nn.Module):
    """
    Selective Deep Integration (U-Net v2 style)
    - inputs: list of feature maps [f1, f2, ..., fN] (each shape: B x C_i x H_i x W_i)
    - out_channels: desired output channel (int)
    - reduction: channel reduction in gate MLP (int)
    """
    def __init__(self, in_channels: List[int], out_channels: int, reduction: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = len(in_channels)

        # per-scale 1x1 conv to unify channels before fuse
        self.proj = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            for c in in_channels
        ])

        # per-scale gate MLPs (channel-wise sigmoid gates)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, max(out_channels // reduction, 1), kernel_size=1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(out_channels // reduction, 1), out_channels, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
            for _ in range(self.n)
        ])

        # fusion conv (after concat)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * self.n, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """
        xs: list of feature maps (length n)
        returns: fused feature map with channels = out_channels and spatial = spatial of xs[-1]
        """
        assert len(xs) == self.n, f"Expect {self.n} feature maps, got {len(xs)}"

        target = xs[-1]
        target_size = target.shape[2:]  # (H, W)

        proj_feats = []
        gated_feats = []

        # project each scale to same channel dim and resize to target spatial
        for i, x in enumerate(xs):
            x_p = self.proj[i](x)  # B x out_ch x Hi x Wi
            if x_p.shape[2:] != target_size:
                x_p = F.interpolate(x_p, size=target_size, mode='bilinear', align_corners=False)
            proj_feats.append(x_p)

            # gate MLP uses pooled representation (mean) -> deterministic-safe
            g = self.gates[i](x_p.mean(dim=(2, 3), keepdim=True))  # B x out_ch x1 x1
            gated_feats.append(x_p * g)

        # weighted sum (selective aggregation)
        sum_f = torch.stack(gated_feats, dim=0).sum(dim=0)  # B x out_ch x H x W

        # concat all projected feats and fuse
        concat_f = torch.cat(proj_feats, dim=1)  # B x (out_ch * n) x H x W
        fused = self.fuse(concat_f)  # B x out_ch x H x W

        # residual-style combine
        out = fused + sum_f

        return out
