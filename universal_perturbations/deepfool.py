import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

"""
Obtain perturbation delta v for a given image
"""

def deepfool(img, classifier, num_classes=10, overshoot=0.02, max_iter=50):
	"""
	@img: 
		Either 1xCxHxW or CxHxW.
	@classifier:
		Returns logits for 10 classes
	@overshoot:
		termination criteria for vanishing gradients
	@max_iter:	
		max iterations for deepfool.
	"""

	f_image = classifier.forward(img).requires_grad_().data.numpy().flatten()
	I = f_image.argsort()[::-1]
	I = I[0:num_classes]
	label = I[0]

	input_shape = img.numpy().shape
	pert_img = copy.deepcopy(img)

	w = np.zeros(input_shape)
	r_total = np.zeros(input_shape)

	x = Variable(pert_img, requires_grad=True)

	loop_i = 0

	fs = classifier.forward(x)
	k_i = label

	while k_i==label and loop_i < max_iter:
		pert = np.inf
		fs[0, I[0]].backward(retain_graph=True)
		og_grad = x.grad.data.numpy().copy()

		for k in range(1, num_classes):
			zero_gradients(x)

			fs[0, I[k]].backward(retain_graph=True)
			curr_grad = x.grad.data.numpy().copy()

			w_k = curr_grad - og_grad
			f_k = (fs[0, I[k]] - fs[0, I[0]]).data.numpy()

			pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

			if pert_k < pert:
				pert = pert_k
				w = w_k

		r_i = (pert+1e-5) * w / np.linalg.norm(w)
		r_total = np.float32(r_i + r_total)

		pert_img = img + (1+overshoot) * torch.from_numpy(r_total)

		x = Variable(pert_img, requires_grad=True)
		fs = classifier.forward(x)
		k_i = np.argmax(fs.data.numpy().flatten())

		loop_i += 1

	return (1+overshoot) * r_total, loop_i, label, k_i, pert_img