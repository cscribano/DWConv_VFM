import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Mapping, Any
import warnings

import os
import sys

sys.path.append('')

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False

class ConvAttention_DW_gumbel(nn.Module):
    def __init__(
            self,
            dim: int,
            input_height: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            cls_token: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            init_conv: bool = False,
            memeff: bool = True
    ) -> None:
        super().__init__()

        self.memeff = memeff and XFORMERS_AVAILABLE

        self.input_height = input_height
        self.cls = cls_token

        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # for both conv and atn!
        self.conv_dim = dim
        self.scale = self.head_dim ** -0.5

        # Note: groups=dim
        self.qk_conv = nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim,
                                 groups=self.conv_dim, kernel_size=3, stride=1, padding=1, bias=qkv_bias)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if init_conv:
            self.init_conv()

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        B, N, C = x.shape

        if self.memeff:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

            q, k, v = unbind(qkv, 2)
            x_attn = memory_efficient_attention(q, k, v, attn_bias=None)
            x_attn = x_attn.reshape([B, N, C])
            v = v.permute(0, 2, 1, 3)

        else:
            qkv = (
                self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            )

            q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

            # Regular attention
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Convolution attention
        d1 = self.input_height
        d2 = N // d1
        v_conv = v.transpose(1, 2).reshape(B, self.conv_dim, d1,
                                           d2).contiguous()  # (B, N, d) -> (B, d, N) -> (B, d, s, s)
        x_conv = self.qk_conv(v_conv).reshape(B, self.conv_dim, N).transpose(1, 2)  # scale?

        # Block-level
        if len(w) == 1:
            # Block-level
            w = w[0]
            x = w * x_conv + (1 - w) * x_attn
        elif len(w) == self.num_heads:
            # Head-level
            w = w.repeat_interleave(self.head_dim).view(1, 1, -1)
            x = w * x_conv + (1 - w) * x_attn
        else:
            raise ValueError(f"w has len {len(w)}, which is not supported")

        # Output projection
        x = self.attn_drop(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def init_conv(self, size=3, sigma=1.0):
        coords = torch.arange(size) - size // 2
        x, y = torch.meshgrid(coords, coords)

        # Compute the 2D Gaussian function
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()  # Normalize to ensure sum equals 1
        kernel = kernel.expand_as(self.qk_conv.weight).clone()
        self.qk_conv.weight = torch.nn.Parameter(kernel)

    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        return super().load_state_dict(state_dict, strict=False)

class ConvAttention_spvit_gumbel(nn.Module):
    def __init__(
            self,
            dim: int,
            input_height: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            cls_token: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            init_conv: bool = False,
            memeff: bool = True
    ) -> None:
        super().__init__()

        self.memeff = memeff and XFORMERS_AVAILABLE

        self.input_height = input_height
        self.cls = cls_token

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # for both conv and atn!
        self.conv_dim = dim
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.alpha = 1e-2
        self.max_kernel_size = 3
        self.register_parameter("head_probs", nn.Parameter(torch.ones([self.num_heads, (self.max_kernel_size ** 2)])))
        self.bn_3x3 = nn.Identity()  # nn.BatchNorm2d(self.head_dim)
        self.conv_act = nn.ReLU()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def prepare_weights(self):

        _, _, vw = torch.chunk(self.qkv.weight, 3)
        _, _, vb = torch.chunk(self.qkv.bias, 3)

        head_probs = (self.head_probs.view(self.num_heads, (self.max_kernel_size ** 2)) / self.alpha).softmax(
            0)  # (Nh, 9)

        new_v_weight = vw.view(self.num_heads, self.head_dim, self.dim).permute(1, 2,
                                                                                0) @ head_probs  # (Hd, dv (1024), 9)
        new_v_bias = (vb.view(self.num_heads, self.head_dim).permute(1, 0) @ head_probs).sum(-1)  # (Hd)
        new_proj_weight = (
                    self.proj.weight.view(self.dim, self.num_heads, self.head_dim).permute(0, 2, 1) @ head_probs).sum(
            -1)  # (dv, Hd)
        kernel_3x3 = new_v_weight.permute(2, 0, 1).view(3, 3, self.head_dim, -1).permute(2, 3, 1,
                                                                                         0)  # (Hd, dv, 3,3 )= (64, 1024, 3, 3)

        return kernel_3x3, new_v_bias, new_proj_weight

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        B, N, C = x.shape

        if self.memeff:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

            q, k, v = unbind(qkv, 2)
            x_attn = memory_efficient_attention(q, k, v, attn_bias=None)
            x_attn = x_attn.reshape([B, N, C])

        else:
            qkv = (
                self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            )

            q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

            # Regular attention
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Convolution attention
        kernel_3x3, new_v_bias, new_proj_weight = self.prepare_weights()

        # The bias for conv is the bias sum of nine heads
        d1 = self.input_height
        d2 = N // d1
        x_conv = x.view(x.shape[0], d1, d2, x.shape[2]).permute(0, 3, 1, 2)  # (B, NhxHd, h, w) = (B, 1024, 24,24)
        x_conv = F.conv2d(x_conv, kernel_3x3, padding=1, bias=new_v_bias)  # (B, Hd, h, w) = (1,64,24,24)
        x_conv = self.conv_act(self.bn_3x3(x_conv)).permute(0, 2, 3, 1).flatten(1, 2)  # (1, N, Hd) = (1, 576, 64)
        x_conv = F.linear(x_conv, new_proj_weight, self.proj.bias)  # 64->1024

        # Block-level
        w = w[0]
        x = w * x_conv + (1 - w) * x_attn

        # Output projection
        x = self.attn_drop(x) 
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        return super().load_state_dict(state_dict, strict=False)
