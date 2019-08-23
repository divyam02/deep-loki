import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

"""
Obtain perturbation delta v for a given image
"""

def deepfool(img, classifier, grads, num_classes=10, overshoot=0.02, max_iter=50):
	"""
	@img: 
		Either 1xCxHxW or CxHxW.
	@classifier:
		Returns logits for 10 classes
	@grads:
		???
	@overshoot:
		termination criteria for vanishing gradients
	@max_iter:	
		max iterations for deepfool.
	"""
	pass