import copy
import importlib
import tensorflow as tf

from .craft_loss_v3 import CRAFTLossV3
from .mask_l1_loss import MaskL1Loss
from .dice_loss import DiceLoss
from .balance_cross_entropy_loss import BalanceCrossEntropyLoss


def build_losses(config):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    losses = []
    for cfg in config:
        name = str(list(cfg.keys())[0])
        value = list(cfg.values())[0]
        coeff = value.pop("coeff")
        arch = getattr(mod, name)(**value)
        losses.append({'loss': arch, 'coeff': coeff})
    return losses
