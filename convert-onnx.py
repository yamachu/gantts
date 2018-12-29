"""Evaluation script for GAN-based VC models.

usage: evaluation_vc.py [options] <checkpoint> <data_dir> <output>

options:
    -h, --help                  Show this help message and exit
    --diffvc                    Enable DIFF VC.
"""
import os
import sys
from os.path import join

import numpy as np
import torch
import torch.onnx
from docopt import docopt
from torch.autograd import Variable

import gantts
from hparams import vc as hp


def load_checkpoint(model, optimizer, checkpoint_path):
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    data_dir = args["<data_dir>"]
    out_path = args["<output>"]

    # Collect stats
    data_mean = np.load(join(data_dir, "data_mean.npy"))
    data_var = np.load(join(data_dir, "data_var.npy"))
    data_std = np.sqrt(data_var)

    if hp.generator_params["in_dim"] is None:
        hp.generator_params["in_dim"] = data_mean.shape[-1]
    if hp.generator_params["out_dim"] is None:
        hp.generator_params["out_dim"] = data_mean.shape[-1]

    # Model
    model = getattr(gantts.models, hp.generator)(**hp.generator_params)
    load_checkpoint(model, None, checkpoint_path)

    x = Variable(torch.randn(1, 1, 177))
    R = Variable(torch.randn(1, 3))

    torch.onnx.export(model, (x, R), out_path, verbose=True)

    sys.exit(0)
