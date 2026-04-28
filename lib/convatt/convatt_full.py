import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
from typing import Mapping, Any, Union
import warnings

import os
import sys

sys.path.append('')

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (H - Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (H - Attention)")

class CnvAttnFull_ensembled(nn.Module): # aka: SPViT
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
            kernel_size: int = 3,
            **kwargs
    ) -> None:
        super().__init__()

        assert qkv_bias is True and proj_bias is True
        self.alpha = 1e-2

        self.input_height = input_height
        self.cls = cls_token
        self.num_heads = num_heads

        self.head_dim = dim // num_heads  # for both conv and atn!
        self.dim = dim

        self.max_kernel_size = kernel_size
        self.register_parameter("head_probs", nn.Parameter(torch.ones([self.num_heads, (self.max_kernel_size ** 2)])))
        self.bn_3x3 = nn.Identity()  # nn.BatchNorm2d(self.head_dim)
        self.conv_act = nn.ReLU()

        self.v = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def prepare_weights(self):
        head_probs = (self.head_probs.view(self.num_heads, (self.max_kernel_size ** 2)) / self.alpha).softmax(
            0)  # (Nh, 9)

        new_v_weight = self.v.weight.view(self.num_heads, self.head_dim, self.dim).permute(1, 2,
                                                                                           0) @ head_probs  # (Hd, dv (1024), 9)
        new_v_bias = (self.v.bias.view(self.num_heads, self.head_dim).permute(1, 0) @ head_probs).sum(-1)  # (Hd)
        new_proj_weight = (
                    self.proj.weight.view(self.dim, self.num_heads, self.head_dim).permute(0, 2, 1) @ head_probs).sum(
            -1)  # (dv, Hd)
        kernel_3x3 = new_v_weight.permute(2, 0, 1).view(3, 3, self.head_dim, -1).permute(2, 3, 1,
                                                                                         0)  # (Hd, dv, 3,3 )= (64, 1024, 3, 3)

        return kernel_3x3, new_v_bias, new_proj_weight

    def forward(self, x: Tensor) -> Tensor:

        B, N, C = x.shape

        kernel_3x3, new_v_bias, new_proj_weight = self.prepare_weights()

        # The bias for conv is the bias sum of nine heads
        d1 = self.input_height
        d2 = N // d1
        x = x.view(x.shape[0], d1, d2, x.shape[2]).permute(0, 3, 1, 2)  # (B, NhxHd, h, w) = (B, 1024, 24,24)
        x = F.conv2d(x, kernel_3x3, padding=1, bias=new_v_bias)  # (B, Hd, h, w) = (1,64,24,24)
        x = self.conv_act(self.bn_3x3(x)).permute(0, 2, 3, 1).flatten(1, 2)  # (1, N, Hd) = (1, 576, 64)
        x = F.linear(x, new_proj_weight, self.proj.bias)  # 64->1024

        return x

    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):

        new_dict = {}

        qw, kw, vw = torch.chunk(state_dict["qkv.weight"], 3)
        new_dict["v.weight"] = vw

        if state_dict.get("qkv.bias", None) is not None:
            qb, kb, vb = torch.chunk(state_dict["qkv.bias"], 3)
            new_dict["v.bias"] = vb

        new_dict["proj.weight"] = state_dict["proj.weight"]
        if state_dict.get("proj.bias", None) is not None:
            new_dict["proj.bias"] = state_dict["proj.bias"]

        return super().load_state_dict(new_dict, strict=False)

class CnvAttnFull_unensembled(nn.Module):
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
            kernel_size: int = 3,
            init_conv: bool = True,
            **kwargs
    ) -> None:
        super().__init__()

        assert qkv_bias is True and proj_bias is True
        self.alpha = 1e-2

        self.input_height = input_height
        self.cls = cls_token
        self.num_heads = num_heads

        self.head_dim = dim // num_heads  # for both conv and atn!
        self.dim = dim

        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.max_kernel_size = kernel_size
        self.register_parameter("head_probs", nn.Parameter(torch.ones([self.num_heads, (self.max_kernel_size ** 2)])))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        if init_conv:
            self.init_conv()

    def prepare_weights(self):

        K = self.max_kernel_size ** 2
        head_probs = self.head_probs.view(self.num_heads, K)

        wp = self.v_proj.weight
        bp = self.v_proj.bias

        wp = wp.view(self.num_heads, self.head_dim, self.dim).unsqueeze(1).expand(-1, K, -1, -1)
        wp = wp * head_probs.unsqueeze(-1).unsqueeze(-1)
        kernel_3x3 = wp.transpose(0, 1).reshape(K, self.dim, self.dim).permute(1,2,0).reshape(self.dim, self.dim,
                                                       self.max_kernel_size, self.max_kernel_size)

        bp = bp.view(self.num_heads, self.head_dim)
        bp = bp.unsqueeze(0).expand(K, -1, -1)
        kernel_bias = (bp * head_probs.transpose(0, 1).unsqueeze(-1)).reshape(K, self.dim).sum(0)

        return kernel_3x3, kernel_bias

    def forward(self, x: Tensor) -> Tensor:

        B, N, C = x.shape

        kernel_3x3, kernel_bias = self.prepare_weights()

        # The bias for conv is the bias sum of nine heads
        d1 = self.input_height
        d2 = N // d1
        x = x.view(x.shape[0], d1, d2, x.shape[2]).permute(0, 3, 1, 2) # (B, NhxHd, h, w) = (B, 1024, 24,24)
        x = F.conv2d(x, kernel_3x3, padding=1, bias=kernel_bias).reshape(B, self.dim, N).transpose(1, 2) # (B, Hd, h, w) = (1,64,24,24)

        # Output projection
        x = self.attn_drop(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):

        new_dict = {}

        qw, kw, vw = torch.chunk(state_dict["qkv.weight"], 3)
        new_dict["v_proj.weight"] = vw

        if state_dict.get("qkv.bias", None) is not None:
            qb, kb, vb = torch.chunk(state_dict["qkv.bias"], 3)
            new_dict["v_proj.bias"] = vb

        new_dict["proj.weight"] = state_dict["proj.weight"]
        if state_dict.get("proj.bias", None) is not None:
            new_dict["proj.bias"] = state_dict["proj.bias"]

        return super().load_state_dict(new_dict, strict=False)

    def init_conv(self, size=3, sigma=1.0):
        coords = torch.arange(size) - size // 2
        x, y = torch.meshgrid(coords, coords)

        # Compute the 2D Gaussian function
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()  # Normalize to ensure sum equals 1
        kernel = kernel.flatten().expand_as(self.head_probs).clone()
        self.head_probs = torch.nn.Parameter(kernel)