import torch
import torch.nn as nn
import torch.transforms as transforms
from deepfool import *
import numpy as np

"""
Functions for network specific images,
creating batches etc.
"""

def project_norm(v, norm_size, p_value):
	"""
	Project perturbation norm onto sphere(0, xi)
	"""
	if p_value==2:
		norm = v * min(1, norm_size/np.linalg.norm(v.flatten(1)))	# v is a what?
	elif p_value==np.inf:
		v = np.sign(v) * np.minimum(abs(v), norm_size)
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

def get_fooling_rate(val_loader, classifier ,perturbation):
	"""
	Calculate fooling rate: 
	got diff answer / all answers
	"""
	v = perturbation
	val_iter = iter(val_loader)
	fooled = 0
	total = list(val_iter.size())[0]

	with torch.no_grad():
		for i in range(total):
			data = next(val_iter)
			img, label = data
			output = classifier(img)
			_, true_predicted = torch.max(output.data, 1)
			output_pertubed = classifier(img+perturbation)
			_, pert_predicted = torch.max(output.data, 1)

			if true_predicted!=pert_predicted:
				fooled+=1

		print("Fooling rate on validation set", fooled/total, "\n")

	return fooled/total

def get_univ_pert(	train_loader, val_loader, classifier, delta=0.1, 
					max_iter=np.inf, norm_size=10, p_value=np.inf, 
					num_classes=10, overshoot=0.02, max_iter_deepfool=10):
	
	"""
	Returns universal perturbation vector.

    @train_loader: 
    	Images of size MxCxHxW (M: number of images), in tensor form.

	@val_loader:
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
    	# Shuffle data!		
		for i in range(train_size):
			curr_img = train_data[i]

			if np.argmax(classifier(curr_img).numpy().flatten()) == np.argmax(classifier(curr_img + v).numpy().flatten()):
				"""
				Get incremental perturbation delta_vi for image i
				"""
				print("Iteration", total_steps)
				delta_vi, iter, _, _, _ = deepfool(curr_img+v, classifier, num_classes=num_classes, 
												overshoot=overshoot, max_iter = max_iter_deepfool)
				if iter < max_iter_deepfool:
					v += delta_vi
					v = project_norm(v, norm_size, p_value)

		total_steps+=1
		fooling_rate = get_fooling_rate(val_loader, classifier, v)

	return v