import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# 1. Select 和 transnext_multi_scale_wrapper
#    从真实实现文件导入，不在此重复定义
# ─────────────────────────────────────────────
from ultralytics.nn.modules.transnext_multi_scale_wrapper import (
    transnext_multi_scale_wrapper,
    Select,
)

# ─────────────────────────────────────────────
# 2. Concat_BiFPN：BiFPN风格通道拼接
# ─────────────────────────────────────────────
class Concat_BiFPN(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


# ─────────────────────────────────────────────
# 3. C2PSA_DAT：带DAT注意力的C2PSA变体
#    保留你自己的实现，此处为占位
# ─────────────────────────────────────────────
class C2PSA_DAT(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.cv = nn.Sequential(
            nn.Conv2d(c1, c2, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        return self.cv(x)


# ─────────────────────────────────────────────
# 4. C3k2_ConvNeXtV2Block：ConvNeXtV2风格的C3k2
#    保留你自己的实现，此处为占位
# ─────────────────────────────────────────────
class C3k2_ConvNeXtV2Block(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, k=1, g=False):
        super().__init__()
        self.cv = nn.Sequential(
            nn.Conv2d(c1, c2, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        return self.cv(x)