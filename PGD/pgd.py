import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
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


def perturb_img(img, label, target, model, step_size=0.5, 
				max_iters=100, epsilon=0.03125, norm='inf'):
	"""
	Perturb image using Projected Gradient Descent...
	@Input: Preprocessed images (1xCxHxW)
	"""
	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	loss_fn = torch.nn.CrossEntropyLoss()
	true_img = get_true_img(img.clone().detach().requires_grad_(True).cuda())
	adv = img.clone().detach().requires_grad_(True).cuda()
	targeted = label==target

	for step in range(max_iters):
		optimizer.zero_grad()
		# temp_adv = adv.clone().detach().requires_grad_(True).cuda()
		pred = model(adv)
		loss = loss_fn(pred, target)
		loss.backward()
		optimizer.step()

		with torch.no_grad():
			# if norm is 'inf':
			# 	grads = step_size * temp_adv.grad.sign()
				
			# if targeted:
			# 	adv -= grads
			# else:
			# 	adv += grads

			if norm is 'inf':
				# Projection on to the l-inf norm ball. We can directly clip errant values...
				# or we can scale it down. As far as the paper goes, clipping is fine.
				true_adv = get_true_img(adv)

				# adv = adv.clamp(true_img-epsilon, true_img+epsilon)
				true_adv = torch.max(torch.min(true_adv, true_img+epsilon), 
								true_img-epsilon)
				true_adv = true_adv.clamp(0, 1)
				preprocessed_true_adv = process_img(true_adv)

				pred = model(preprocessed_true_adv)

				# print("iteration:", step+1, "Loss:", loss)
				_, temp = torch.max(pred, 1)
				if temp==target:
					# print("Success!")
					break
			else:
				raise NotImplementedError

			# if (step+1)%10==0:

	return preprocessed_true_adv