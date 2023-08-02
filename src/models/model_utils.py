from __future__ import absolute_import

from src.models.transformer import Transformer
from src.models.transformer_vit16 import Transformer_ViT16


def create_model(cfg):
    model_type = cfg.model.model_type
    if model_type == "transformer":
        model = Transformer()
    elif model_type == "transformer_vit16":
        model = Transformer_ViT16()    
    return model
