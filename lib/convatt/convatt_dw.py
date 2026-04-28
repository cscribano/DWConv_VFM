import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
from typing import Mapping, Any, Union

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

class CnvAttnDW_blockwise(nn.Module):
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
            **kwargs
    ) -> None:
        super().__init__()

        self.input_height = input_height
        self.cls = cls_token

        self.head_dim = dim // num_heads  # for both conv and atn!
        self.conv_dim = dim
        self.scale = self.head_dim ** -0.5

        # Note: groups=dim
        self.qk_conv = nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim,
                                 groups=self.conv_dim, kernel_size=3, stride=1, padding=1, bias=qkv_bias)

        # Cls params
        if self.cls:
            self.cls_all = nn.Parameter(torch.rand(num_heads))
            self.all_cls = nn.Parameter(torch.rand(num_heads))

        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if init_conv:
            self.init_conv()

    def forward(self, x: Tensor) -> Tensor:

        # B, N, C = x.shape
        B = int(x.size(0))  # Explicitly get batch size
        N = int(x.size(1))
        C = int(x.size(2))

        v = self.v_proj(x)

        if self.cls:
            d1 = self.input_height
            d2 = (N - 1) // d1
            v_cls = v[:, 0]
            v_conv = v[:, 1:].transpose(1, 2).reshape(B, self.conv_dim, d1, d2)  # (B, N, d) -> (B, d, N) -> (B, d, s, s)

            # CLS token integration (w/ parameter sharing)
            x_cls = v_cls + self.cls_all.repeat_interleave(self.head_dim).unsqueeze(0) * v_conv.mean((-1, -2))
            v_conv = v_conv + rearrange((self.all_cls.repeat_interleave(self.head_dim).unsqueeze(0) * v_cls),
                                        "b n -> b n 1 1")

            x_conv = self.qk_conv(v_conv).reshape(B, self.conv_dim, N - 1).transpose(1, 2)  # scale?
            x = torch.cat([x_cls.unsqueeze(1), x_conv], dim=1)
        else:
            d1 = self.input_height
            d2 = N // d1
            v_conv = v.transpose(1, 2).reshape(B, self.conv_dim, d1,
                                               d2).contiguous()  # (B, N, d) -> (B, d, N) -> (B, d, s, s)
            x = self.qk_conv(v_conv).reshape(B, self.conv_dim, N).transpose(1, 2)  # scale?

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


class CnvAttnDW_scatter(nn.Module): # HAttention
    def __init__(
            self,
            dim: int,
            input_height: int,
            num_heads: int = 8,
            conv_idx: Union[list[int], None] = None,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            cls_token: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            memeff: bool = False,
            init_conv: bool = False
    ) -> None:
        super().__init__()

        self.memeff = memeff and XFORMERS_AVAILABLE
        self.input_height = input_height

        conv_heads = 0 if conv_idx is None else len(conv_idx)
        # Attention heads first, then conv heads
        if conv_idx is None:
            conv_idx = []

        self.perm = conv_idx + [i for i in range(num_heads) if i not in conv_idx]
        assert len(self.perm) == num_heads, (self.perm, conv_idx)
        self.cls = cls_token

        self.conv_heads = conv_heads
        assert conv_heads <= num_heads, print(conv_heads)
        self.atn_heads = num_heads - conv_heads

        self.head_dim = dim // num_heads  # for both conv and atn!
        self.conv_dim = self.head_dim * self.conv_heads
        self.scale = self.head_dim ** -0.5

        if self.conv_heads > 0:
            # Note: groups=dim
            self.qk_conv = nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim,
                                     groups=self.conv_dim, kernel_size=3, stride=1, padding=1, bias=qkv_bias)

            # Cls params
            if self.cls:
                self.cls_all = nn.Parameter(torch.rand(self.conv_heads))
                self.all_cls = nn.Parameter(torch.rand(self.conv_heads))

        if self.atn_heads > 0:
            self.lin_dim = self.head_dim * self.atn_heads
            self.q_proj = nn.Linear(dim, self.lin_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, self.lin_dim, bias=qkv_bias)

        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if init_conv:
            self.init_conv()

    def forward(self, x: Tensor) -> Tensor:

        # B, N, C = x.shape
        B = int(x.size(0))  # Explicitly get batch size
        N = int(x.size(1))
        C = int(x.size(2))

        v = self.v_proj(x)

        x_out = []
        if self.conv_heads > 0:
            # Convolutional heads
            if self.cls:
                d1 = self.input_height
                d2 = (N - 1) // d1
                v_cls = v[:, 0, :self.conv_dim]
                v_conv = v[:, 1:, :self.conv_dim].transpose(1, 2).reshape(B, self.conv_dim, d1,
                                                                          d2)  # (B, N, d) -> (B, d, N) -> (B, d, s, s)

                # CLS token integration (w/ parameter sharing)
                x_cls = v_cls + self.cls_all.repeat_interleave(self.head_dim).unsqueeze(0) * v_conv.mean((-1, -2))
                v_conv = v_conv + rearrange((self.all_cls.repeat_interleave(self.head_dim).unsqueeze(0) * v_cls),
                                            "b n -> b n 1 1")

                x_conv = self.qk_conv(v_conv).reshape(B, self.conv_dim, N - 1).transpose(1, 2)  # scale?
                x_out.append(torch.cat([x_cls.unsqueeze(1), x_conv], dim=1))
            else:
                d1 = self.input_height
                d2 = N // d1
                v_conv = v[:, :, :self.conv_dim].transpose(1, 2).reshape(B, self.conv_dim, d1,
                                                                         d2).contiguous()  # (B, N, d) -> (B, d, N) -> (B, d, s, s)
                x_conv = self.qk_conv(v_conv).reshape(B, self.conv_dim, N).transpose(1, 2)  # scale?
                x_out.append(x_conv)

        if self.atn_heads > 0:
            # Attention heads
            if self.memeff:
                q = self.q_proj(x).reshape(B, N, self.atn_heads, self.head_dim)
                k = self.k_proj(x).reshape(B, N, self.atn_heads, self.head_dim)
                v_atn = v[..., self.conv_dim:].reshape(B, N, self.atn_heads, self.head_dim)

                x_atn = memory_efficient_attention(q, k, v_atn)
                x_atn = x_atn.reshape([B, N, self.atn_heads * self.head_dim])
            else:
                q = self.q_proj(x).reshape(B, N, self.atn_heads, self.head_dim).transpose(1,
                                                                                          2) * self.scale  # B, Nh, N, dq
                k = self.k_proj(x).reshape(B, N, self.atn_heads, self.head_dim).transpose(1, 2)  # B, Nh, N, dk
                v_atn = v[..., self.conv_dim:].reshape(B, N, self.atn_heads, self.head_dim).transpose(1,
                                                                                                      2)  # B, Nh, N, dv

                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                # attn = self.attn_drop(attn)

                # x_atn = (attn @ v_atn).transpose(2,3).reshape(B, self.atn_heads*self.head_dim, N).transpose(1,2)
                x_atn = (attn @ v_atn).transpose(1, 2).reshape(B, N, self.atn_heads * self.head_dim)

            x_out.append(x_atn)

        # Output projection
        x = torch.cat(x_out, dim=-1)
        x = self.attn_drop(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def _rearrange_weights(self, T, permutation, d, mode):
        # Rearrange weights on loading

        assert mode in ["r", "c", "a"]
        M = T.shape[0]  # Number of rows in T

        # Initialize an empty array to hold the rearranged tensor
        rearranged = torch.zeros_like(T)

        # Fill the rearranged tensor according to the permutation
        for i, p in enumerate(permutation):
            start_idx = p * d
            end_idx = (p + 1) * d
            if mode == "r":
                rearranged[i * d:(i + 1) * d, ...] = T[start_idx:end_idx, ...]
            elif mode == "c":
                rearranged[..., i * d:(i + 1) * d] = T[..., start_idx:end_idx]
            else:
                rearranged[i * d:(i + 1) * d] = T[start_idx:end_idx]

            # print(f"({start_idx}:{end_idx}) -> ({i * d}:{(i + 1) * d})")

        return rearranged

    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):

        new_dict = {}

        qw, kw, vw = torch.chunk(state_dict["qkv.weight"], 3)
        qw = self._rearrange_weights(qw, self.perm, self.head_dim, mode="r")
        kw = self._rearrange_weights(kw, self.perm, self.head_dim, mode="r")
        vw = self._rearrange_weights(vw, self.perm, self.head_dim, mode="r")

        if self.atn_heads > 0:
            new_dict["q_proj.weight"] = qw[self.conv_dim:]
            new_dict["k_proj.weight"] = kw[self.conv_dim:]
            new_dict["v_proj.weight"] = vw

        if state_dict.get("qkv.bias", None) is not None:
            qb, kb, vb = torch.chunk(state_dict["qkv.bias"], 3)
            qb = self._rearrange_weights(qb, self.perm, self.head_dim, mode="a")
            kb = self._rearrange_weights(kb, self.perm, self.head_dim, mode="a")
            vb = self._rearrange_weights(vb, self.perm, self.head_dim, mode="a")

            if self.atn_heads > 0:
                new_dict["q_proj.bias"] = qb[self.conv_dim:]
                new_dict["k_proj.bias"] = kb[self.conv_dim:]
                new_dict["v_proj.bias"] = vb

        pw = state_dict["proj.weight"]
        pw = self._rearrange_weights(pw, self.perm, self.head_dim, mode="c")
        new_dict["proj.weight"] = pw

        if state_dict.get("proj.bias", None) is not None:
            pb = state_dict["proj.bias"]
            new_dict["proj.bias"] = pb

        return super().load_state_dict(new_dict, strict=False)

    def init_conv(self, size=3, sigma=1.0):
        coords = torch.arange(size) - size // 2
        x, y = torch.meshgrid(coords, coords)

        # Compute the 2D Gaussian function
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()  # Normalize to ensure sum equals 1
        kernel = kernel.expand_as(self.qk_conv.weight).clone()
        self.qk_conv.weight = torch.nn.Parameter(kernel)


class CnvAttnDW_blockwise_ensembled(nn.Module):
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

        self.max_kernel_size = 1
        self.register_parameter("head_probs", nn.Parameter(torch.ones([self.num_heads, (self.max_kernel_size ** 2)])))
        self.bn_3x3 = nn.Identity()  # nn.BatchNorm2d(self.head_dim)
        self.conv_act = nn.ReLU()

        self.v = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        self.v_dw_conv = nn.Conv2d(self.head_dim, self.head_dim, 3, 1, 1, groups=self.head_dim, bias=True)

    def prepare_weights(self):
        head_probs = (self.head_probs.view(self.num_heads, (self.max_kernel_size ** 2)) / self.alpha).softmax(
            0)  # (Nh, 9)

        new_v_weight = self.v.weight.view(self.num_heads, self.head_dim, self.dim).permute(1, 2,
                                                                                           0) @ head_probs  # (Hd, dv (1024), 9)
        new_v_weight = new_v_weight.squeeze(-1)
        new_v_bias = (self.v.bias.view(self.num_heads, self.head_dim).permute(1, 0) @ head_probs).sum(-1)  # (Hd)
        # kernel_1x1 = new_v_weight.permute(2, 0, 1).view(1, 1, self.head_dim, -1).permute(2, 3, 1, 0)  # (Hd, dv, 3,3 )= (64, 1024, 3, 3)

        new_proj_weight = (
                    self.proj.weight.view(self.dim, self.num_heads, self.head_dim).permute(0, 2, 1) @ head_probs).sum(
            -1)  # (dv, Hd)

        return new_v_weight, new_v_bias, new_proj_weight

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape

        new_v_weight, new_v_bias, new_proj_weight = self.prepare_weights()
        x = F.linear(x, new_v_weight, new_v_bias)  # (B, Hd, h, w) = (1,64,24,24)

        # The bias for conv is the bias sum of nine heads
        d1 = self.input_height
        d2 = N // d1
        x = x.view(B, d1, d2, self.head_dim).permute(0, 3, 1, 2).contiguous()  # (B, NhxHd, h, w) = (B, 1024, 24,24)
        x = self.v_dw_conv(x)
        # x = self.conv_act(self.bn_3x3(x)).permute(0, 2, 3, 1).flatten(1, 2) # (1, N, Hd) = (1, 576, 64)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)  # (1, N, Hd) = (1, 576, 64) # <<< NO RELU!!
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