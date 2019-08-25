import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import copy
import matplotlib.pyplot as plt

"""
Obtain perturbation delta v for a given image
"""

def side_plot(og_img, pert_img, label, k_i):
	"""
	Quick visual debug!
	"""
	_, ax = plt.subplots(2)

	classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	#og_img = inv_transform(og_img)
	og_img[0][0] *= 0.2023
	og_img[0][1] *= 0.1994
	og_img[0][2] *= 0.2010
	og_img[0][0] += 0.4914
	og_img[0][1] += 0.4822
	og_img[0][2] += 0.4465

	pert_img[0][0] *= 0.2023
	pert_img[0][1] *= 0.1994
	pert_img[0][2] *= 0.2010
	pert_img[0][0] += 0.4914
	pert_img[0][1] += 0.4822
	pert_img[0][2] += 0.4465

	#pert_img = inv_transform(pert_img)

	ax[0].imshow(og_img.data.cpu().squeeze().permute(1, 2, 0))
	ax[1].imshow(pert_img.data.cpu().squeeze().permute(1, 2, 0))
	ax[0].set_title(classes[label])
	ax[1].set_title(classes[k_i])
	plt.savefig('works.png')

def deepfool(img, original_image, classifier, num_classes=10, overshoot=0.002, max_iter=50):
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
	img = img.cuda()
	classifier = classifier.cuda()
	f_image = classifier.forward(img).requires_grad_().data.cpu().numpy().flatten()
	I = f_image.argsort()[::-1]
	I = I[0:num_classes]
	label = I[0]
	input_shape = img.cpu().numpy().shape
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
		og_grad = x.grad.data.cpu().numpy().copy()

		for k in range(1, num_classes):
			zero_gradients(x)
			fs[0, I[k]].backward(retain_graph=True)
			curr_grad = x.grad.data.cpu().numpy().copy()
			w_k = curr_grad - og_grad
			f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

			pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

			if pert_k < pert:
				pert = pert_k
				w = w_k
				#print("pert", pert_k)

		r_i = (pert+1e-5) * w / np.linalg.norm(w)
		r_total = np.float32(r_i + r_total)

		#r_total = torch.from_numpy(r_total).float().cuda()

		pert_img = img + (1+overshoot) * torch.from_numpy(r_total).float().cuda()
		x = Variable(pert_img, requires_grad=True)
		fs = classifier.forward(x)
		k_i = np.argmax(fs.data.cpu().numpy().flatten())

		#if k_i!=label:
		#	side_plot(original_image, pert_img, label, k_i)

		#print(k_i, label)

		loop_i += 1

	return (1+overshoot) * torch.from_numpy(r_total).float().cuda(), loop_i, label, k_i