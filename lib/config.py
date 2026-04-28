import os

from enum import Enum
from pathlib import Path
from typing import Optional, Union, Any, Literal
from typing_extensions import Annotated
from pydantic import BaseModel, field_validator, model_validator, Field

IS_EXPORT = os.getenv("CONF_EXPORT_MODE", "false").lower() == "true"

class BackboneType(str, Enum):
    dinov2 = "dinov2"
    mae = "mae"
    clip = "clip"

class ConvType(str, Enum):
    dw = "depthwise"
    full = "full"
    nop = "nop"

class ConvGranularity(str, Enum):
    blockwise = "blockwise"
    scattered = "scattered"

class ConvSelection(str, Enum):
    dsp = "dsp"
    manual = "manual"

class ConvAttention(BaseModel):
    # Convolutional attention plugin
    conv_style: ConvType = "depthwise"
    conv_granularity: ConvGranularity = "blockwise"
    conv_selection: ConvSelection = "manual"
    head_ensembling: bool = False

    heads: Optional[dict[int, list[int]]] = None
    blocks: Optional[list[int]] = None

    gumbel_w_init: Optional[list[float]] = None
    gumbel_k: Optional[int] = None
    gates_lr: Optional[float] = None
    gates_wd: Optional[float] = 0.0
    gates_lr_fixed: Optional[bool] = False
    tau: Optional[list] = None
    tau_decay_n: Optional[int] = None

    weight_decay: bool
    layer_decay: bool
    memeff: Optional[bool] = None
    conv_init: bool = False
    nop: bool = False
    freeze_model: bool = False

    class Config:
        use_enum_values = True

    @model_validator(mode='before')
    @classmethod
    def validate_heads(cls, values):

        heads = values.get('heads', None)
        if heads is not None:
            conv_style = values.get("conv_style")
            assert conv_style == ConvType.dw, "Head-level selection is only supported for classic dw conv"

        gumbel = values.get('conv_selection', 'manual') == ConvSelection.dsp
        if gumbel:
            assert values.get('gumbel_k', None) is not None, "Provide gumbel_k when using gumbel"
            assert values.get('heads', None) is None, "heads is incompatible with gumbel"
            assert values.get('blocks', None) is None, "blocks is incompatible with gumbel"
            tau = values.get('tau', None)
            assert tau is not None, "Provide tau, either [fixed value] or [tau_start, tau_end]"
            assert len(tau) in [1, 2], "Provide tau, either [fixed value] or [tau_start, tau_end]"
            if len(tau) == 2:
                assert values.get('tau_decay_n', None) is not None, "Missing tau_decay_n"

            conv_style = values.get("conv_style")
            assert conv_style in [ConvType.full, ConvType.dw], \
                "Only classic and spvit style conv layer are supported with gumbel serch"

        return values

class Backbone(BaseModel):
    backbone: BackboneType
    pretrained_path: Path | None = None
    backbone_pretrained_path: Path | None
    output_layers: list[int]
    freeze: bool = False

    @field_validator("backbone_pretrained_path")
    def validate_backbone_checkpoint(cls, backbone_pretrained_path: Path | None) -> Path | None:
        if backbone_pretrained_path is None:
            return None
        if not backbone_pretrained_path.exists():
            raise ValueError(f"Pretrained MAE checkpoint {backbone_pretrained_path} does not exist")
        return backbone_pretrained_path

    class Config:
        use_enum_values = True


class ViTBackbone(Backbone):
    # ViT patch size
    backbone: Literal[BackboneType.dinov2, BackboneType.mae, BackboneType.clip]
    patch_size: int
    embed_dim: int
    depth: int | None
    num_heads: int | None
    mlp_ratio: float
    drop_path_rate: float
    include_cls_token: bool
    pretrained_img_size: int
    learnable_pos_embed: bool
    include_task_token: bool = False
    num_register_tokens: int | None

class Config(BaseModel):
    backbone: ViTBackbone = Field(discriminator="backbone")  # `backbone` uses enum
    float32_matmul_precision: str
    fp16: bool
    img_size: int
    convatt: Optional[ConvAttention] = None

    @model_validator(mode='after')
    def validate_convatt(self):

        if self.convatt is None:
            return self

        if self.convatt.heads is not None:
            block_idx = list(self.convatt.heads.keys())  # type: list[int]
            if max(block_idx) >= self.backbone.depth or min(block_idx) < 0:
                raise ValueError("Specified convatt block is not valid")

            heads_idx_all = list(self.convatt.heads.values())
            for head_idx in heads_idx_all:
                if not len(head_idx) == len(set(head_idx)):
                    raise ValueError(f"Duplicate convatt head in {head_idx}")
                if max(head_idx) >= self.backbone.num_heads or min(head_idx) < 0:
                    raise ValueError(f"Specified convatt head is not valid in {head_idx}")

        if self.convatt.blocks is not None:
            block_idx = self.convatt.blocks
            if max(block_idx) >= self.backbone.depth or min(block_idx) < 0:
                raise ValueError("Specified convatt block is not valid")
            
        return self