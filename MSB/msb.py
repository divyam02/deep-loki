import torch
import torch.nn as nn
import numpy as np


def add_msb_noise(net, img, fraction, to_numpy=False):
	"""
	Select three sets of pixels randomly across all channels.
	Not sure about how to control perturbation size...
	Shouldn't be much, this is only a 3x32x32 pixel image...

	Cap L2 norm of distance for now.
	"""

	pert_img = torch.clone(img)
	