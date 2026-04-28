import os
import torch
import torch.nn as nn

from lib.models.layers.attention import Attention
from lib.models.layers.patch_embed import PatchEmbed
from lib.convatt.convatt_full import CnvAttnFull_ensembled, CnvAttnFull_unensembled
from lib.convatt.convatt_dw import CnvAttnDW_blockwise_ensembled
from lib.convatt.dsp import ConvAttention_DW_gumbel, ConvAttention_spvit_gumbel

class ExportPatchEmbed(PatchEmbed):
    def forward(self, x):
        #_, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        """assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"""

        x = self.proj(x)  # B C H W
        H, W = int(x.size(2)), int(x.size(3))
        # EXPORT MOD
        #x = x.flatten(2).transpose(1, 2)  # B HW C
        x = x.permute(0,2,3,1).reshape(-1, H*W, self.embed_dim)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x


@torch.no_grad()
def inference_export(self, x):
    latents = self.encode(x, None)
    output_embeds = self.decoder(latents, None, self.config.img_size)  # (B, dec_embed_dim, H, W)

    preds = {}
    for k in self.heads.keys():
        pred = self.heads[k](output_embeds)  # (B, task_out_channels, H, W)
        preds[k] = pred

    return preds


def full_infer_patch(m: CnvAttnFull_unensembled) -> None:
    kernel_3x3, kernel_bias = m.prepare_weights()

    m.v_conv = nn.Conv2d(m.dim, m.dim, 3, 1, 1, bias=True)

    m.v_conv.weight.data = kernel_3x3
    m.v_conv.bias.data = kernel_bias

    # clean
    del m.v_proj
    del m.head_probs

    def forward(self, x):
        B, N, C = x.shape

        # The bias for conv is the bias sum of nine heads
        d1 = self.input_height
        d2 = N // d1
        x = x.view(x.shape[0], d1, d2, x.shape[2]).permute(0, 3, 1, 2) # (B, NhxHd, h, w) = (B, 1024, 24,24)
        x = self.v_conv(x) # (B, Hd, h, w) = (1,64,24,24)
        x = x.permute(0, 2, 3, 1).flatten(1, 2) # (1, N, Hd) = (1, 576, 64)
        x = self.proj(x) # 64->1024

        return x

    # patch forward
    m.forward = forward.__get__(m)

def spv_infer_patch(m: CnvAttnFull_ensembled) -> None:

    kernel_3x3, new_v_bias, new_proj_weight = m.prepare_weights()

    m.v_conv = nn.Conv2d(m.dim, m.head_dim, 3, 1, 1, bias=True)
    m.new_proj = nn.Linear(m.head_dim, m.dim, bias=True)

    m.v_conv.weight.data = kernel_3x3
    m.v_conv.bias.data = new_v_bias

    m.new_proj.weight.data = new_proj_weight
    m.new_proj.bias.data = m.proj.bias.data

    # clean
    del m.proj
    del m.v
    del m.head_probs

    def forward(self, x):
        B, N, C = x.shape

        # The bias for conv is the bias sum of nine heads
        d1 = self.input_height
        d2 = N // d1
        x = x.view(x.shape[0], d1, d2, x.shape[2]).permute(0, 3, 1, 2) # (B, NhxHd, h, w) = (B, 1024, 24,24)
        x = self.v_conv(x) # (B, Hd, h, w) = (1,64,24,24)
        x = self.conv_act(self.bn_3x3(x)).permute(0, 2, 3, 1).flatten(1, 2) # (1, N, Hd) = (1, 576, 64)
        x = self.new_proj(x) # 64->1024

        return x

    # patch forward
    m.forward = forward.__get__(m)

def spv_pwdw_infer_patch(m: CnvAttnDW_blockwise_ensembled) -> None:

    new_v_weight, new_v_bias, new_proj_weight = m.prepare_weights()

    m.new_v_proj = nn.Linear(m.dim, m.head_dim, bias=True)
    m.new_proj = nn.Linear(m.head_dim, m.dim, bias=True)

    m.new_v_proj.weight.data = new_v_weight
    m.new_v_proj.bias.data = new_v_bias

    m.new_proj.weight.data = new_proj_weight
    m.new_proj.bias.data = m.proj.bias.data

    # clean
    del m.proj
    del m.v
    del m.head_probs

    def forward(self, x):
        B, N, C = x.shape

        x = self.new_v_proj(x) # v

        # The bias for conv is the bias sum of nine heads
        d1 = self.input_height
        d2 = N // d1
        x = x.view(x.shape[0], d1, d2, m.head_dim).permute(0, 3, 1, 2)
        x = self.v_dw_conv(x)
        # x = self.conv_act(self.bn_3x3(x)).permute(0, 2, 3, 1).flatten(1, 2)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        x = self.new_proj(x)

        return x

    # patch forward
    m.forward = forward.__get__(m)


def apply_patches(model: nn.Module, input_shape: tuple, output_layers: list[int]):

    h, w = input_shape

    # Patch attention modules
    for module in model.modules():
        if isinstance(module, Attention):
            pass
        elif isinstance(module, CnvAttnFull_unensembled):
            full_infer_patch(module)
        elif isinstance(module, CnvAttnDW_blockwise_ensembled):
            spv_pwdw_infer_patch(module)
        elif isinstance(module, CnvAttnFull_ensembled):
            spv_infer_patch(module)
        elif isinstance(module, ConvAttention_DW_gumbel) or isinstance(module, ConvAttention_spvit_gumbel):
            raise NotImplementedError("Export patch for DSP-based methods has not been implemented yet!")
        elif isinstance(module, PatchEmbed):
            module.__class__ = ExportPatchEmbed

    # Patch PE
    M = (h // model.patch_size) * (w // model.patch_size)
    x = torch.empty((1, M, model.embed_dim), dtype=torch.float32)
    pe = model.interpolate_pos_encoding(x, h, w).detach()

    def fixed_pe(self, x, w, h):
        return pe

    model.interpolate_pos_encoding = fixed_pe.__get__(model)

    # Patch forward function
    @torch.no_grad()
    def inference_bacbone_only(self, x):
        latents = self.get_intermediate_layers(
            x, output_layers, return_class_token=False
        )
        return latents

    model.forward = inference_bacbone_only.__get__(model)