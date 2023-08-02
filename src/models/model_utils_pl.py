from __future__ import absolute_import

from src.models.transformer_pl import Transformer_pl

def create_model(conf, args):
    model_type = conf.model.model_type
    print("Creating model: ", model_type)
    if model_type == "transformer":
        model = Transformer_pl(conf, args)
    elif model_type == "transformer_vit16":
        model = Transformer_pl(conf, args)
    return model
