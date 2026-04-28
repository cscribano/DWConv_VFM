import sys
sys.path.append('..')

import os
os.environ["CONF_EXPORT_MODE"] = "true"
os.environ["XFORMERS_DISABLED"] = "true"

import torch
from omegaconf import OmegaConf

from lib.config import Config
from lib.deploy.export_patches import apply_patches
from lib.models.dino_vision_transformer import DinoVisionTransformer
from lib.patching.patch import patch_dinov2_convatt

def main(conf_file):

    # Load Config File
    model_config = OmegaConf.load(conf_file)
    cfg = Config(**model_config)  # type: ignore

    # Load Model and weights
    model = DinoVisionTransformer(
        img_size=cfg.backbone.pretrained_img_size,
        patch_size=cfg.backbone.patch_size,
        embed_dim=cfg.backbone.embed_dim,
        depth=cfg.backbone.depth,
        num_heads=cfg.backbone.num_heads,
        mlp_ratio=cfg.backbone.mlp_ratio,
        drop_path_rate=cfg.backbone.drop_path_rate,
        num_register_tokens=cfg.backbone.num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0,
        block_chunks=0,
        init_values=1e-5,  # I don't know the correct value, but it doesn't matter. It is loaded with the checkpoint
        use_cls=cfg.backbone.include_cls_token
    )

    model = model.eval()

    # Apply conv attention
    patch_dinov2_convatt(model, cfg)
    model = model.to('cuda')

    # TODO: Load your final checkpoint (with convolution weights)
    weights = ...
    torch.load(weights, map_location='device')

    # Apply inference patches
    apply_patches(model, (cfg.img_size, cfg.img_size), cfg.backbone.output_layers)

    # Export ( no dynamic batch)
    input = torch.rand((1,3, cfg.img_size, cfg.img_size), dtype=torch.float32).cuda()
    torch.onnx.export(model, (input),
                      "model.onnx",
                      opset_version=17,
                      input_names=['input'],
                      )


if __name__ == '__main__':
    main("configs/dw_12_blockwise.yaml")