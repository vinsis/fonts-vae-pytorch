import torch
import torch.nn as nn

def get_loss(x_generated, x, mu, logvar):
    bce = nn.BCELoss(reduction='sum')(x_generated, x)
    # bce = nn.BCELoss()(x_generated, x) * 40
    kldiv = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kldiv
