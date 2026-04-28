import torch
from omegaconf import OmegaConf

from lib.config import Config
from lib.models.dino_vision_transformer import DinoVisionTransformer
from lib.patching.patch import patch_dinov2_convatt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(conf_file):

    # Load the YAML configuration
    base_config = OmegaConf.load(conf_file)

    # load_model model
    cfg = Config(**base_config)  # type: ignore
    backbone = DinoVisionTransformer(
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

    print("[ MODEL LOADED]")

    # TODO: Load your finetuned checkpoint BEFORE applying convolutional patches
    """
    finetuned_weights = ...
    torch.load(finetuned_weights, map_location=device)
    """

    # Apply patches
    patch_dinov2_convatt(backbone, cfg)
    backbone.eval()
    backbone = backbone.to(device)

    # Dummy forward
    imgs = torch.rand(1, 3, cfg.img_size, cfg.img_size).to(device)
    outputs = backbone.get_intermediate_layers(
        imgs, cfg.backbone.output_layers, return_class_token=False
    )

    # TODO : Implement your Finetuning loop here


if __name__ == '__main__':
    main("configs/full_12_ensemble_dsp.yaml")