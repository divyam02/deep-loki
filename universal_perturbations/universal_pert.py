import torch
import torch.nn as nn
import torch.transforms as transforms
from deepfool import *
import numpy as np

"""
Functions for network specific images,
creating batches etc.
"""

def project_norm(v, xi, p):
	"""
	Project perturbation norm onto sphere(0, xi)
	"""
	if p==2:
		norm = v * min(1, xi/np.linalg.norm(v.flatten(1)))	# v is a what?
	elif p==np.inf:
		v = np.sign(v) * np.minimum(abs(v), xi)
	else:
		raise ValueError("Projection unavailable for given norm value.")

	return v

def tensor2numpy(train_data):
	"""
	returns list of numpy arrays
	"""
	new_numpy
	for i in train_data:
		new_numpy.append(i.numpy())

	return new_numpy

def get_fooling_rate(val_data, perturbation):
	"""
	Calculate fooling rate: 
	val_data accuracy/ perturbed_val_data accuracy
	"""
	v = perturbation
	pass

def universal_perturbation(	train_data, val_data,classifier, grads, delta=0.1, 
							max_iter=np.inf, norm_size=10, p_value=np.inf, 
							num_classes=10, overshoot=0.02, max_iter_deepfool=10):
	
	"""
    @train_data: 
    	Images of size MxCxHxW (M: number of images), in tensor form.

	@val_data:
		images of siz MxCxHxW (M: number of images), in tensor form. Use for fooling rate calculation!
 
    @classifier: 
    	feedforward function (input: images, output: values of activation BEFORE softmax).

    @grads: 
    	gradient functions with respect to input (as many gradients as classes).

    @delta: 
    	controls the desired fooling rate (default = 80% fooling rate)

    @max_iter: 
    	optional other termination criterion (maximum number of iteration, default = np.inf)

    @norm_size: 
    	controls the l_p magnitude of the perturbation (default = 10)

    @p_value: 
    	norm to be used (2, inf, default = np.inf)

    @num_classes: 
    	num_classes (limits the number of classes to test against, by default = 10)

    @overshoot: 
    	used as a termination criterion to prevent vanishing updates (default = 0.02).

    @max_iter_deepfool: 
    	maximum number of iterations for deepfool (default = 10)

    return: the universal perturbation.
    """
    v = 0 # perturbation vector 
    fooling_rate = 0.0
    total_steps = 0
    #np_train_data = tensor2numpy(train_data)
    train_size = list(np_train_data.size())[0]

    while fooling_rate<1-delta and total_steps<max_iter:
    	# Shuffle data
		
		for i in range(train_size):
			curr_img = train_data[i]

			if np.argmax(classifier(curr_img).numpy().flatten()) == np.argmax(classifier(curr_img + v).numpy().flatten()):
				"""
				Get incremental perturbation delta_vi for image i
				"""
				pass

		total_steps+=1

