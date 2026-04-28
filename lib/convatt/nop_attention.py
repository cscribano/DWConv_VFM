import torch
import torch.nn as nn
from torch import Tensor
from typing import Mapping, Any
import sys
sys.path.append('.')


class NopAttention(nn.Module):
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

        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:

        # Output projection
        x = self.proj(x)
        x = self.drop(x)

        return x

    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):

        new_dict = {}

        _, _, vw = torch.chunk(state_dict["qkv.weight"], 3)
        pw = state_dict["proj.weight"]
        new_w = torch.mm(pw, vw)
        new_dict["proj.weight"] = new_w

        new_b = None
        if state_dict.get("qkv.bias", None) is not None:
            _, _, vb = torch.chunk(state_dict["qkv.bias"], 3)
            new_b = torch.matmul(pw, vb)

        if state_dict.get("proj.bias", None) is not None:
            pb = state_dict["proj.bias"]

            if new_b is not None:
                new_b = new_b + pb
            else:
                new_b = pb

        if new_b is not None:
            new_dict["proj.bias"] = new_b

        return super().load_state_dict(new_dict, strict=False)
