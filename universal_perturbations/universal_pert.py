import torch
import torch.nn as nn
import torchvision.transforms as transforms
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
		#v = np.sign(v) * np.minimum(abs(v), torch.tensor(norm_size))
		v = torch.sign(v) * torch.min(abs(v), torch.tensor(norm_size).cuda())
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
	total = 20
	print("Validation data length:", total)
	with torch.no_grad():
		for i in range(total):
			data = next(val_iter)
			img, label = data
			img = img.cuda()

			output = classifier(img)
			_, true_predicted = torch.max(output.data, 1)

			output_pertubed = classifier(img + v)
			_, pert_predicted = torch.max(output_pertubed.data, 1)

			#print(true_predicted, pert_predicted)

			fooled += (true_predicted != pert_predicted).sum().item()

		print("Fooling rate on validation set", fooled/5000, fooled)

	return fooled/5000

def get_univ_pert(	train_loader, val_loader, classifier, is_cuda, delta=0.15, 
					max_iter=np.inf, norm_size=0.5, p_value=np.inf, 
					num_classes=10, overshoot=0.02, max_iter_deepfool=15):
	
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
	train_size  = 45000
	train_iter = iter(train_loader)
	while fooling_rate < 1-delta and total_steps < max_iter:
		# Shuffle data!	
		if total_steps==train_size:
			train_iter = iter(train_loader)

		print("Total passes:", total_steps)	
		for i in range(train_size):
			curr_img, label = next(train_iter)
			if is_cuda:
				curr_img = curr_img.cuda()
			if np.argmax(classifier(curr_img).detach().cpu().numpy().flatten()) == np.argmax(classifier(curr_img + v).detach().cpu().numpy().flatten()):
				"""
				Get incremental perturbation delta_vi for image i
				"""
				#print("images read", i+1, "\n")
				#print(curr_img)
				#input()
				delta_vi, d_iter, new_label, true_label = deepfool(curr_img+v, curr_img, classifier, num_classes=num_classes, 
												overshoot=overshoot, max_iter = max_iter_deepfool)

				#print("iters needed:", d_iter)
				if d_iter < max_iter_deepfool:
					v += delta_vi
					v = project_norm(v, norm_size, p_value)

			if (i+1)%2500 == 0:
				fooling_rate = get_fooling_rate(val_loader, classifier, v)
				if fooling_rate >= 1 - delta:
					break
				else:
					print("processed:", i+1)
					print("perturbation", v, v.size(), "\n")

		total_steps+=1
		fooling_rate = get_fooling_rate(val_loader, classifier, v)

	side_plot(curr_img, curr_img+v, new_label, true_label)

	return v