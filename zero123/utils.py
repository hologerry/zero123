import argparse
import datetime
import glob
import os
import sys
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from parser_helpers import get_parser, nondefault_trainer_args
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.distributed import rank_zero_only


MULTINODE_HACKS = False


@rank_zero_only
def rank_zero_print(*args):
    print(*args)


def modify_weights(w, scale=1e-6, n=2):
    """Modify weights to accomodate concatenation to unet"""
    extra_w = scale * torch.randn_like(w)
    new_w = w.clone()
    for i in range(n):
        new_w = torch.cat((new_w, extra_w.clone()), dim=1)
    return new_w
