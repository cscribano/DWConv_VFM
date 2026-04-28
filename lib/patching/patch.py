import torch
from torch import Tensor, nn

from lib.config import Config, ConvType, ConvGranularity, ConvSelection
from lib.convatt.convatt_dw import CnvAttnDW_blockwise, CnvAttnDW_scatter, CnvAttnDW_blockwise_ensembled
from lib.convatt.convatt_full import CnvAttnFull_ensembled, CnvAttnFull_unensembled
from lib.convatt.dsp import ConvAttention_DW_gumbel, ConvAttention_spvit_gumbel
from lib.convatt.nop_attention import NopAttention
from lib.patching.gumbel import gumbel_topk, TauAnealing

from lib.models.layers import MemEffAttention
from lib.models.layers.block import drop_add_residual_stochastic_depth


def patch_dinov2_convatt(backbone, config):

    CLASSES_DICT = {
        (ConvType.nop, ConvGranularity.blockwise, False, ConvSelection.manual): NopAttention,

        (ConvType.dw, ConvGranularity.scattered, False, ConvSelection.manual): CnvAttnDW_scatter,
        (ConvType.dw, ConvGranularity.blockwise, False, ConvSelection.manual): CnvAttnDW_blockwise,
        (ConvType.dw, ConvGranularity.blockwise, True, ConvSelection.manual): CnvAttnDW_blockwise_ensembled,

        (ConvType.full, ConvGranularity.blockwise, True, ConvSelection.manual): CnvAttnFull_ensembled, # SPViT
        (ConvType.full, ConvGranularity.blockwise, False, ConvSelection.manual): CnvAttnFull_unensembled,

        (ConvType.dw, ConvGranularity.scattered, False, ConvSelection.dsp): ConvAttention_DW_gumbel,
        (ConvType.dw, ConvGranularity.blockwise, False, ConvSelection.dsp): ConvAttention_DW_gumbel,
        (ConvType.full, ConvGranularity.blockwise, True, ConvSelection.dsp): ConvAttention_spvit_gumbel,
    }

    if config.convatt is None:
        return

    # Blockwise heads
    convatt_blocks = config.convatt.blocks
    if convatt_blocks is None:
        convatt_blocks = []

    # Scattered heads
    convatt_heads = config.convatt.heads
    if convatt_heads is not None:
        convatt_blocks += list(config.convatt.heads.keys())

    # Add weights for DSP gates
    if config.convatt.conv_selection == ConvSelection.dsp:
        if config.convatt.conv_granularity == ConvGranularity.scattered: # scattered
            backbone.gates = nn.Parameter(torch.empty(config.backbone.depth * config.backbone.num_heads))
        else: # Blockwise
            backbone.gates = nn.Parameter(torch.empty(config.backbone.depth))

        if config.convatt.gumbel_w_init is not None:
            assert len(config.convatt.gumbel_w_init) == len(backbone.gates)
            backbone.gates.data = torch.tensor(config.convatt.gumbel_w_init)
        else:
            nn.init.uniform_(backbone.gates)

        # Monkey patching, replace forward function
        backbone.gumbel_k = config.convatt.gumbel_k
        if len(config.convatt.tau) == 2:
            # Tau with anealing function
            ti, te = config.convatt.tau
            backbone.tau_fn = TauAnealing(ti, te, config.convatt.tau_decay_n).tau
        else:
            # Constant tau
            backbone.tau_fn = lambda _: config.convatt.tau[0]

        backbone._get_intermediate_layers_not_chunked = get_intermediate_layers_not_chunked_gumbel.__get__(backbone)

        # Patch all the blocks
        convatt_blocks = [i for i in range(config.backbone.depth)]

    # Patch attention Layers
    for i, block in enumerate(backbone.blocks):
        if i in convatt_blocks:
            old_att = block.attn

            att_cls = CLASSES_DICT.get((config.convatt.conv_style,
                                       config.convatt.conv_granularity,
                                       config.convatt.head_ensembling,
                                       config.convatt.conv_selection), None)

            if att_cls is None:
                raise ValueError(f"Invalid convatt: style: {config.convatt.conv_style}, granularity: {config.convatt.conv_granularity}, ensembling: {config.convatt.head_ensembling}, selection method: {config.convatt.conv_selection}")
            else:
                print(f"[OK] Patching attention block {i} with class {str(att_cls)}:"
                      f" style: {config.convatt.conv_style}, granularity: {config.convatt.conv_granularity}, "
                      f"ensembling: {config.convatt.head_ensembling}, selection method: {config.convatt.conv_selection}")

            if isinstance(att_cls, CnvAttnDW_scatter):
                new_att = att_cls(
                    dim=config.backbone.embed_dim,
                    input_height=config.img_size // backbone.patch_size,
                    num_heads=config.backbone.num_heads,
                    conv_idx=config.convatt.heads[i],
                    qkv_bias=old_att.qkv.bias is not None,
                    proj_bias=old_att.proj.bias is not None,
                    attn_drop=old_att.attn_drop.p,
                    proj_drop=old_att.proj_drop.p,
                    cls_token=config.backbone.include_cls_token,
                    init_conv=config.convatt.conv_init,
                    memeff=isinstance(old_att, MemEffAttention)
                )
            else:
                new_att = att_cls(
                    dim=config.backbone.embed_dim,
                    input_height=config.img_size // backbone.patch_size,
                    num_heads=config.backbone.num_heads,
                    qkv_bias=old_att.qkv.bias is not None,
                    proj_bias=old_att.proj.bias is not None,
                    attn_drop=old_att.attn_drop.p,
                    proj_drop=old_att.proj_drop.p,
                    cls_token=config.backbone.include_cls_token,
                    init_conv=config.convatt.conv_init,
                    memeff=isinstance(old_att, MemEffAttention)
                )

            if config.convatt.conv_selection == ConvSelection.dsp:
                block.forward = block_forward_gumbel.__get__(block)

            # Recover weights
            old_weights = old_att.state_dict()
            mk, uk = new_att.load_state_dict(old_weights)
            assert len(uk) == 0, print(mk, uk)
            # assert len(mk) == 4 #['cls_all', 'all_cls', 'qk_conv.weight', 'qk_conv.bias']

            # swap
            backbone.blocks[i].attn = new_att
            del old_att # make sure


def get_intermediate_layers_not_chunked_gumbel(self, x, n=1):

    # sample gumbel vector of w...
    if self.training:
        self.tau = self.tau_fn(self.current_epoch)
        gumbel_w = gumbel_topk(self.gates, self.gumbel_k, self.tau)
    else:
        topk_ind = torch.topk(self.gates, self.gumbel_k).indices
        gumbel_w = torch.zeros_like(self.gates).scatter(0, topk_ind, 1)

    gumbel_w = gumbel_w.reshape((len(self.blocks), -1))

    x = self.prepare_tokens_with_masks(x)
    # If n is an int, take the n last blocks. If it's a list, take them
    output, total_block_len = [], len(self.blocks)
    blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
    for i, blk in enumerate(self.blocks):
        x = blk(x, gumbel_w[i])
        if i in blocks_to_take:
            output.append(x)
    assert len(output) == len(
        blocks_to_take
    ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
    return output


def block_forward_gumbel(self, x: Tensor, w: float) -> Tensor:
    def attn_residual_func(x: Tensor) -> Tensor:
        return self.ls1(self.attn(self.norm1(x), w))

    def ffn_residual_func(x: Tensor) -> Tensor:
        return self.ls2(self.mlp(self.norm2(x)))

    if self.training and self.sample_drop_ratio > 0.1:
        # the overhead is compensated only for a drop path rate larger than 0.1
        x = drop_add_residual_stochastic_depth(
            x,
            residual_func=attn_residual_func,
            sample_drop_ratio=self.sample_drop_ratio,
        )
        x = drop_add_residual_stochastic_depth(
            x,
            residual_func=ffn_residual_func,
            sample_drop_ratio=self.sample_drop_ratio,
        )
    elif self.training and self.sample_drop_ratio > 0.0:
        x = x + self.drop_path1(attn_residual_func(x))
        x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
    else:
        x = x + attn_residual_func(x)
        x = x + ffn_residual_func(x)
    return x