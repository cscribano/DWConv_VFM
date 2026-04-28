import os
import sys
sys.path.append('../..')
import torch
from omegaconf import OmegaConf

import random
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

os.environ["CONF_EXPORT_MODE"] = "true"
os.environ["XFORMERS_DISABLED"] = "true"

import logging
from argparse import ArgumentParser

from tqdm import tqdm
#from deploy.torch_to_onnx import load_model
from deploy.export_patches import inference_export

from lib.config import Config
from lib.models.layers.attention import Attention
from lib.models.dino_vision_transformer import DinoVisionTransformer

log = logging.getLogger(__name__)

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

class Stats:
    def __init__(self):
        
        self.n = 0
        self.running_mean = None
        self.running_M2 = None

    def __call__(self, module, module_in, module_out):

        batch = module.attn_weights.detach().cpu()
        self.update(batch)

    def update(self, batch):
        B = batch.shape[0]  # Batch size, Width, Height
        batch_size = B

        if self.running_mean is None:
            # Initialize mean and M2 on the first batch
            self.running_mean = torch.zeros_like(batch.mean(dim=0), dtype=torch.float64)
            self.running_M2 = torch.zeros_like(batch.mean(dim=0), dtype=torch.float64)

        delta = batch - self.running_mean
        self.running_mean += torch.sum(delta, dim=0) / (self.n + batch_size)
        self.running_M2 += torch.sum(delta * (batch - self.running_mean), dim=0)
        self.n += batch_size

    @property
    def mean(self):
        return self.running_mean
    
    @property
    def std(self):
        if self.n > 1:
            variance = self.running_M2 / (self.n - 1)  # Bessel's correction
            variance = torch.clamp(variance, min=0)  # Ensure no negative values
        else:
            variance = torch.zeros_like(self.running_M2)  # Avoid division by zero
        std_dev = torch.sqrt(variance)

        return std_dev

def main():

    parser = ArgumentParser()
    parser.add_argument("--conf_file", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run_name", type=str)

    args = parser.parse_args()

    set_all_seeds(69)

    # Load Config File
    model_config = OmegaConf.load(args.conf_file)
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

    # Load weights
    ck = torch.load(args.checkpoint)
    model.load_state_dict(ck["model"], strict=True)
    model = model.to("cuda")

    # Patch forward function
    model.forward = inference_export.__get__(model)

    # Load and configure dataset
    train_data = ... # TODO: Instantiate your dataset implementation here!

    ss_ind = random.sample(range(len(train_data)), 1000)
    train_data = torch.utils.data.Subset(train_data, ss_ind)
    loader = DataLoader(train_data, batch_size=8, num_workers=8)

    stat_classes = {}

    # Register hooks to capture attention maps for specific layers
    for layer_id, layer in enumerate(model.backbone.blocks):  # Adjust for ViT structure

        # Patch attention to collect
        if isinstance(layer.attn, Attention):
            print("Hooking attention layer")
            layer.attn.__class__ = AttentionHookable

            layer.attn.layer_id = layer_id  # Optional: add an identifier to each layer

            sc = Stats()
            stat_classes[layer_id] = sc
            layer.attn.register_forward_hook(sc)

    feat_dir = Path(__file__).parent / args.run_name
    feat_dir.mkdir(exist_ok=True)

    # Collect attention activation statistics
    for batch in tqdm(loader):  # +1 to handle the last partial batch (if any)
        
        # Inference
        with torch.no_grad():
            _ = model(batch.to("cuda"))

    # Visualize 1st head of 1st layer
    for layer, layer_stats in stat_classes.items():

        layer_mean = layer_stats.mean
        layer_std = layer_stats.std

        torch.save(layer_mean, feat_dir / f"dino_{layer}_mean.pt")
        torch.save(layer_std, feat_dir / f"dino_{layer}_std.pt")

class AttentionHookable(torch.nn.Module):
    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        self.attn_weights = attn #copy.deepcopy(attn)

        #attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == "__main__":
    main()