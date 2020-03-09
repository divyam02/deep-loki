import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_wavelets import DWTForward, DWTInverse
import pywt
import pywt.data
from torchvision import transforms as transforms
import torch.optim as optim
import copy

def perturb_img(img, label, target, model, attack_type, dataset=None, net_type=None, alpha=0.999,
				max_iters=400, epsilon=0.03125, random_restarts=False):
	"""
	Assume learned filters are available.
	"""
	if random_restarts: num_restarts = 5
	else: num_restarts = 1
	prev_count = 0

	best_adv = None
	curr_pred = None

	for restart in range(num_restarts):
		random_init = False
		print('Restart {}'.format(restart+1))
		# Perform DWT on the unnormalized image. Ensure it is done on a copy.
		img.requires_grad = False
		ifm = DWTInverse(mode='zero', wave='db3').cuda()
		ifm.requires_grad = True
		xfm = DWTForward(J=3, mode='zero', wave='db3').cuda()
		xfm.requires_grad = True
		# print(img)
		# input('continue')
		LL, Y = get_dwt_2d(postprocess(img.clone().detach(), dataset, net_type), xfm, dataset)

		# Enable updating of LL, LH, HL, HH.
		LL.requires_grad_(True).cuda()
		Y[0].requires_grad_(True).cuda()
		Y[1].requires_grad_(True).cuda()
		Y[2].requires_grad_(True).cuda()

		optimizer = optim.Adam((LL, Y[0], Y[1], Y[2]), lr=1e-3)

		cls_loss_fn = torch.nn.CrossEntropyLoss()
		dst_loss_fn = torch.nn.PairwiseDistance()

		for step in range(max_iters):

			optimizer.zero_grad()
			# Obtain unnormalized adversarial image by 
			# using IDWT on LL, LH, HL, HH.
			adv = get_inv_dwt_2d(LL, Y, ifm, dataset)

			# print(adv)
			# input('continue?')
			if random_init:
				if dataset=='multi-pie' and net_type!='inception':
					adv += torch.tensor(np.random.normal(-8.0, 8.0, img.clone().detach().cpu().numpy().shape)).cuda()
				else:	
					adv += torch.tensor(np.random.normal(-epsilon, epsilon, img.clone().detach().cpu().numpy().shape)).cuda()
				random_init = False

			# Preprocess image for the network.
			adv = preprocess(clip_pixels(adv, dataset, net_type), dataset, net_type)
			pred = model(adv)

			# print(adv)
			# input('continue')			
			# Control loss tradeoffs with alpha.
			if attack_type=='max_error':
				loss = -1*(1-alpha)*cls_loss_fn(pred, target) + alpha*(torch.mean(dst_loss_fn(torch.flatten(adv, 1, -1), 
																	torch.flatten(img, 1, -1))**2))	
			else:
				loss = (1-alpha)*cls_loss_fn(pred, target) + alpha*(torch.mean(dst_loss_fn(torch.flatten(adv, 1, -1), 
																	torch.flatten(img, 1, -1))**2))

			loss.backward()
			optimizer.step()

			# Check adversary after updating image.
			with torch.no_grad():
				adv = get_inv_dwt_2d(LL, Y, ifm, dataset)
				adv = preprocess(clip_pixels(adv, dataset, net_type), dataset, net_type)
				pred = model(adv)
				# print('pred', torch.max(pred, 1)[1])
				# print('label', label)
				# input('continue?')
				if not attack_type=='max_error':
					if best_adv != None:
						for i in range(len(best_adv)):
							if curr_pred[i]!=target[i]:
								best_adv[i] = adv[i]
					else:
						best_adv = adv.clone()
						curr_pred = torch.max(pred, 1)[1].clone()

				else:
					if best_adv != None:
						for i in range(len(best_adv)):
							if curr_pred[i]==target[i]:
								best_adv[i] = adv[i]
					else:
						best_adv = adv.clone()
						curr_pred = torch.max(pred, 1)[1].clone()


	return best_adv

def preprocess(img, dataset, net_type):
	"""
	Preprocess by normalization. 

	Single image only.
	"""
	if net_type=='inception':
		img = (img - 127.5)/128.0
	
	elif dataset=='multi-pie':
		img[:, 0] -= 131.0912
		img[:, 1] -= 103.8827
		img[:, 2] -= 91.4953
		img[:, 0] /= 1.0
		img[:, 1] /= 1.0
		img[:, 2] /= 1.0

	elif dataset=='fmnist':
		return img

	elif dataset=="tinyimagenet":
		img[:, 0] -= 0.4802
		img[:, 1] -= 0.4481
		img[:, 2] -= 0.3975
		img[:, 0] /= 0.2302
		img[:, 1] /= 0.2265
		img[:, 2] /= 0.2262
	else:
		img[:, 0] -= 0.4914
		img[:, 1] -= 0.4822
		img[:, 2] -= 0.4465
		img[:, 0] /= 0.2023
		img[:, 1] /= 0.1994
		img[:, 2] /= 0.2010

	return img

def postprocess(img, dataset, net_type):
	"""
	Postprocess by unnormalizing

	Single image only.
	"""
	if net_type=='inception':
		img = (img * 128.0) + 127.5
	
	elif dataset=='multi-pie':		
		img[:, 0] *= 1.0
		img[:, 1] *= 1.0
		img[:, 2] *= 1.0
		img[:, 0] += 131.0912
		img[:, 1] += 103.8827
		img[:, 2] += 91.4953

	elif dataset=='fmnist':
		return img
	elif dataset=='tinyimagenet':
		img[:, 0] *= 0.2302
		img[:, 1] *= 0.2265
		img[:, 2] *= 0.2262
		img[:, 0] += 0.4802
		img[:, 1] += 0.4481
		img[:, 2] += 0.3975
	else:
		img[:, 0] *= 0.2023
		img[:, 1] *= 0.1994
		img[:, 2] *= 0.2010
		img[:, 0] += 0.4914
		img[:, 1] += 0.4822
		img[:, 2] += 0.4465	

	return img

def clip_pixels(adv, dataset, net_type):
	"""
	Project img into 0-255 pixel range, based on norm. 

	Simply clamping inside the 0-1 range used in image representations of 
	tensors for the time being.
	"""
	# true_img = postprocess(img.clone().detach())
	# true_adv = torch.max(torch.min(adv, true_img+epsilon), true_img-epsilon)
	if net_type=='inception':
		true_adv = adv.clamp(0, 1)
	elif dataset=='multi-pie':
		true_adv = adv.clamp(0, 255)
	else:
		true_adv = adv.clamp(0, 1)
	return true_adv

def get_inv_dwt_2d(LL, Y, ifm, dataset):
	"""
	Return idwt of bands.

	@ Attribute Y: The entire 3 stacked fine to coarse filter outputs
	"""
	# if dataset=="multi-pie":
	# 	return 255 * ifm((LL, Y))
	return ifm((LL, Y))

def get_dwt_2d(img_batch, xfm, dataset):
	"""
	Return img dwt.
	
	Can probably improve runtimes by not creating a new DWT object per iter.
	"""
	# if dataset=="multi-pie":
	# 	img_batch/=255
	LL, Y = xfm(img_batch)
	return LL, Y