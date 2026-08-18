"""
AggAttn -- standalone, plug-and-play TransNeXt "aggregated attention" block.

Extracted from TransNeXt.py's `AggregatedAttention` class (window-local +
pooled-global hybrid attention, sr_ratio > 1), together with its
ConvolutionalGLU MLP, DropPath, and relative-position-bias machinery, and
repackaged as a single spatial tensor-in / tensor-out module: input
(B, C, H, W) -> output (B, C, H, W), with no internal downsampling.

Differences from the original TransNeXt.py implementation, required to make
this safe to drop into an arbitrary YOLO backbone/neck stage:
  1. Relative position tables are computed lazily from the ACTUAL runtime
     H, W on first forward call and cached per (H, W, device), instead of
     being baked in at construction time from a fixed img_size assumption.
  2. The pooled-branch spatial size (pool_H, pool_W) is likewise computed
     from the actual runtime H, W and sr_ratio at forward time via
     F.adaptive_avg_pool2d, instead of a fixed-size nn.AdaptiveAvgPool2d
     built at construction from a fixed input_resolution. The original code
     silently breaks (or produces a degenerate (0, 0) pool) if the module is
     applied to a feature map smaller than the resolution it was built for.
  3. A 1x1 conv + BN projects in_channels -> out_channels when they differ.

Suggested yaml usage (Ultralytics-style; in_channels auto-injected by
parse_model, so only out_channels onward needs to be listed):
    - [-1, 1, AggAttn, [256, 2, 8, 3, 8, 8.0]]
      # out_channels=256, depth=2, num_heads=8, window_size=3, sr_ratio=8, mlp_ratio=8.0

sr_ratio must leave a non-degenerate pooled grid given the runtime H, W
(roughly sr_ratio <= min(H, W)); prefer StdAttn instead for very small,
deep feature maps (e.g. a stride-64 P6 level), matching TransNeXt's own
choice of sr_ratio=1 at its deepest, smallest-resolution stage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath


@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None):
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)

    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw, indexing='ij')
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw, indexing='ij')

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)
    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table


@torch.no_grad()
def get_seqlen_and_mask(input_resolution, window_size):
    attn_map = F.unfold(torch.ones([1, 1, input_resolution[0], input_resolution[1]]), window_size,
                        dilation=1, padding=(window_size // 2, window_size // 2), stride=1)
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AggregatedAttention(nn.Module):
    """Window-local + pooled-global hybrid attention (TransNeXt's sr_ratio > 1 branch)."""

    def __init__(self, dim, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert window_size % 2 == 1, "window size must be odd"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sr_ratio = sr_ratio
        self.window_size = window_size
        self.local_len = window_size ** 2

        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.temperature = nn.Parameter(torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Pooled-branch components. Pool output size is computed dynamically
        # in forward() from the actual runtime H, W -- see module docstring.
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0, std=0.0004))

        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def _pool_size(self, H, W):
        return max(1, H // self.sr_ratio), max(1, W // self.sr_ratio)

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        B, N, C = x.shape
        pool_H, pool_W = self._pool_size(H, W)
        pool_len = pool_H * pool_W

        total_seq_len = self.local_len + pool_len
        seq_length_scale = torch.log(torch.tensor(total_seq_len, device=x.device, dtype=torch.float32))

        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * seq_length_scale

        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)

        _, padding_mask = get_seqlen_and_mask((H, W), self.window_size)
        padding_mask = padding_mask.to(x.device)

        attn_local = ((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2)
                      + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(padding_mask, float('-inf'))

        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = F.adaptive_avg_pool2d(self.act(self.sr(x_)), (pool_H, pool_W)).reshape(B, -1, pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        kv_pool = self.kv(x_).reshape(B, pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[
            :, relative_pos_index.view(-1)].view(-1, N, pool_len)

        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)
        x_local = (((q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(-2)
                   @ v_local.transpose(-2, -1)).squeeze(-2)
        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AggAttnBlock(nn.Module):
    """One pre-norm residual block: AggregatedAttention + ConvolutionalGLU."""

    def __init__(self, dim, num_heads=8, window_size=3, sr_ratio=8, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AggregatedAttention(dim, num_heads=num_heads, window_size=window_size,
                                        qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                        sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos_index, relative_coords_table))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class AggAttn(nn.Module):
    """
    Plug-and-play spatial-tensor AggAttn module for a YOLO backbone/neck.
    Input (B, in_channels, H, W) -> output (B, out_channels, H, W); H, W are
    unchanged. Stacks `depth` AggAttnBlocks at `out_channels` width.
    """

    def __init__(self, in_channels, out_channels=None, depth=2, num_heads=8, window_size=3,
                 sr_ratio=8, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        out_channels = out_channels or in_channels
        assert out_channels % num_heads == 0, \
            f"out_channels {out_channels} must be divisible by num_heads {num_heads}"

        self.in_proj = (nn.Identity() if out_channels == in_channels else
                        nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels)))
        self.blocks = nn.ModuleList([
            AggAttnBlock(dim=out_channels, num_heads=num_heads, window_size=window_size,
                        sr_ratio=sr_ratio, mlp_ratio=mlp_ratio, drop=drop,
                        attn_drop=attn_drop, drop_path=drop_path)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(out_channels)

        self.sr_ratio = sr_ratio
        # Lazily-populated cache: (H, W, device) -> (relative_pos_index, relative_coords_table).
        self._pos_cache = {}

    def _get_position_tables(self, H, W, device):
        key = (H, W, device)
        cached = self._pos_cache.get(key)
        if cached is not None:
            return cached
        pool_H = max(1, H // self.sr_ratio)
        pool_W = max(1, W // self.sr_ratio)
        relative_pos_index, relative_coords_table = get_relative_position_cpb(
            query_size=(H, W), key_size=(pool_H, pool_W))
        relative_pos_index = relative_pos_index.to(device)
        relative_coords_table = relative_coords_table.to(device)
        self._pos_cache[key] = (relative_pos_index, relative_coords_table)
        return relative_pos_index, relative_coords_table

    def forward(self, x):
        x = self.in_proj(x)
        B, C, H, W = x.shape
        relative_pos_index, relative_coords_table = self._get_position_tables(H, W, x.device)

        x = x.flatten(2).transpose(1, 2)  # B,C,H,W -> B,N,C
        for blk in self.blocks:
            x = blk(x, H, W, relative_pos_index, relative_coords_table)
        x = self.norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = AggAttn(in_channels=256, out_channels=256, depth=2, num_heads=8,
               window_size=3, sr_ratio=8).to(device).eval()
    for hw in [(160, 160), (80, 80), (40, 40)]:
        x = torch.rand(2, 256, *hw).to(device)
        with torch.no_grad():
            y = m(x)
        print(f"AggAttn: input {tuple(x.shape)} -> output {tuple(y.shape)}")
