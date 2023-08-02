from __future__ import absolute_import

import torch

def create_loss_function(cfg):
    loss_type = cfg.loss.loss_type
    if loss_type == "bce_loss":
        criterion = torch.nn.BCELoss()
    return criterion
