import torch
import torch.nn as nn
import torchvision.transforms as transforms__class__
import torchvision.datasets as datasets
import numpy as np
import torch.optim as optim

source_output_activation = list()
guide_output_activation = list()
layer_output_gradient = list()

def visualize_activations(perturbed_feat_map, img_feat_map, guide_feat_map):
	"""
	Display/save matplotlib images of the source and 
	guide. Not much, but a visualization is always 
	nice.
	"""
	raise NotImplementedError

def get_target_layer_named(net, target_layer):
	raise NotImplementedError

def source_forward_hook(self, input, output):
	"""
	Are the gradients required?
	"""
	source_output_activation.append(output)

def guide_forward_hook(self, input, output):
	"""
	Are the gradients required?
	"""
	guide_forward_hook.append(output)

def back_hook(self, input, output):
	"""
	Are gradients needed?
	"""
	layer_output_gradient.append(output)

def perturb_img(img, guide, net, iters=500, lr=1e-3):
	"""
	1. Register hooks for a particular layer.
	2. Run forward pass in eval mode.

	Note: works for modules only!
	"""
	img.requires_grad_()
	# target_layer = get_target_layer_named(net, target_layer)
	# A default way for now... Let's see some results first!
	# I will be trying for reducing representation distance at
	# layer 4 between the guide and at layers 0, 1, 2, 3 between
	# the source.
	net.layer1.register_forward_hook(source_forward_hook)
	net.layer2.register_forward_hook(source_forward_hook)
	net.layer3.register_forward_hook(source_forward_hook)
	net.layer4.register_forward_hook(source_forward_hook)

	net.layer1.register_forward_hook(guide_forward_hook)
	net.layer2.register_forward_hook(guide_forward_hook)
	net.layer3.register_forward_hook(guide_forward_hook)
	net.layer4.register_forward_hook(guide_forward_hook)

	net.layer1.register_backward_hook(back_hook)
	net.layer2.register_backward_hook(back_hook)
	net.layer3.register_backward_hook(back_hook)
	net.layer4.register_backward_hook(back_hook)

	optimizer = optim.Adam([img], lr)	
	for i in range(iters):
		net.forward(img)
		
		net.forward(guide)

	print(img_layer_rep)
	print(guide_layer_rep)	