import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def get_true_img(img):
	"""
	Return img in [0, 1] range.
	"""
	img[0][0] *= 0.2023
	img[0][1] *= 0.1994
	img[0][2] *= 0.2010
	img[0][0] += 0.4914
	img[0][1] += 0.4822
	img[0][2] += 0.4465

	return img

def process_img(img):
	"""
	Perform preprocessing again...
	"""
	img[0][0] -= 0.4914
	img[0][1] -= 0.4822
	img[0][2] -= 0.4465
	img[0][0] /= 0.2023
	img[0][1] /= 0.1994
	img[0][2] /= 0.2010

	return img


def perturb_img(img, label, target, model, step_size=0.3, 
				max_iters=100, epsilon=0.03125, norm='inf'):
	"""
	Perturb image using Projected Gradient Descent...
	@Input: Preprocessed images (1xCxHxW)
	"""
	loss_fn = torch.nn.CrossEntropyLoss()
	true_img = get_true_img(img.clone().detach())
	adv = img.clone().cuda().requires_grad_()
	targeted = label==target

	for step in range(step_size):
		temp_adv = adv.clone().cuda().requires_grad_()
		pred = model(temp_adv)
		loss = loss_fn(pred, target)
		loss.backward()

		if (step+1)%10==0:
			print("iteration:", step+1)

			if pred==target:
				print("Success!")
				break

		with torch.no_grad():
			if norm is 'inf':
				grads = step_size * temp_adv.grad.sign()
				
			if targeted:
				adv -= grads
			else:
				adv += grads

			if norm is 'inf':
				# Projection on to the l-inf norm ball.
				# We can directly clip errant values...
				# or we can scale it down. As far as
				# the paper goes, clipping is fine.
				adv = get_true_img(adv)
				adv = adv.clamp(true_img-epsilon, true_img+epsilon)
				adv = adv.clamp(0, 1)
				adv = process_img(adv)
			else:
				raise NotImplementedError

	return adv