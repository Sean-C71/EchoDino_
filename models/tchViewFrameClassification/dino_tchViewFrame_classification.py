from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import Config
from models.dinov3 import DinoVisionTransformer, vit_large


def getPretrainedDinoEncoder(
        ckpt_path=Config.DINO_DEFAULT_PATH
):
    vit_kwargs = dict(
        img_size=256,
        patch_size=16,
        pos_embed_rope_base=100,
        pos_embed_rope_min_period=None,
        pos_embed_rope_max_period=None,
        pos_embed_rope_normalize_coords='separate',
        pos_embed_rope_shift_coords=None,
        pos_embed_rope_jitter_coords=None,
        pos_embed_rope_rescale_coords=2,
        qkv_bias=True,
        layerscale_init=1.0e-05,
        norm_layer='layernorm',
        ffn_layer='mlp',
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_cls_and_patch_norms=False,
        untie_global_and_local_cls_norm=False,
        device='cpu',
    )
    model = vit_large(**vit_kwargs)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if 'teacher' in ckpt:
            # my weights
            ckpt = ckpt['teacher']
            ckpt = {k.replace('backbone.', ''): v for k, v in ckpt.items() if not 'head' in k}
            model.load_state_dict(ckpt, strict=True)
        else:
            # original weights
            model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


class Dino_TchViewFrame_classification(torch.nn.Module):
    def __init__(self, ckpt_path=Config.DINO_DEFAULT_PATH,
                 train_encoder=False,
                 class_num=56,
                 dim=1024
                 ):
        super().__init__()

        # frame extractor
        backbone_model = getPretrainedDinoEncoder(ckpt_path)
        Config.logger.info(f"Loaded DINO encoder from {ckpt_path}")
        if not train_encoder:
            backbone_model.requires_grad_(False)

        # decoder
        classifier = nn.Linear(dim, class_num)

        # compose
        self.backbone_model = backbone_model
        self.classifier = classifier

    def forward(self, inputs):
        feature = self.backbone_model(inputs)['x_norm_clstoken']  # B*1024
        logits = self.classifier(feature)  # B*C
        return logits

    def getFeatures(self, inputs):
        feature = self.backbone_model(inputs)['x_norm_clstoken']  # B*1024
        return feature


if __name__ == '__main__':
    model = Dino_TchViewFrame_classification(
        ckpt_path=Config.DINO_DEFAULT_PATH, )

    dummy_input = torch.randn(8, 3, 256, 256)
    output = model(dummy_input)

    print(f"Output Shape: {output.shape}")  # should be 8*56
