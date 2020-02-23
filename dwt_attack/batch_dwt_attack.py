import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_wavelets import DWTForward, DWTInverse
import pywt
import pywt.data
from torchvision import transforms as transforms
import torch.optim as optim
import copy

def perturb_img(img, label, target, model, attack_type, dataset=None, alpha=0.5,
				max_iters=300, epsilon=0.03125, random_restarts=False):
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
		ifm.requires_grad = False
		xfm = DWTForward(J=3, mode='zero', wave='db3').cuda()
		xfm.requires_grad = False
		LL, Y = get_dwt_2d(postprocess(img.clone().detach(), dataset), xfm)

		# Enable updating of LL, LH, HL, HH.
		LL.requires_grad_(True).cuda()
		Y[0].requires_grad_(True).cuda()
		Y[1].requires_grad_(True).cuda()
		Y[2].requires_grad_(True).cuda()

		optimizer = optim.Adam((LL, Y[0], Y[1], Y[2]), lr=5e-2)

		cls_loss_fn = torch.nn.CrossEntropyLoss()
		dst_loss_fn = torch.nn.PairwiseDistance()

		for step in range(max_iters):

			optimizer.zero_grad()
			# Obtain unnormalized adversarial image by 
			# using IDWT on LL, LH, HL, HH.
			adv = get_inv_dwt_2d(LL, Y, ifm)
			if not random_init:
				if dataset=='multi-pie':
					adv += torch.tensor(np.random.normal(-8.0, 8.0, img.clone().detach().cpu().numpy().shape)).cuda()
				else:	
					adv += torch.tensor(np.random.normal(-epsilon, epsilon, img.clone().detach().cpu().numpy().shape)).cuda()
				random_init = True

			# Preprocess image for the network.
			adv = preprocess(clip_pixels(adv, dataset), dataset)
			# print('step', step+1)
			pred = model(adv)

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
				adv = get_inv_dwt_2d(LL, Y, ifm)
				adv = preprocess(clip_pixels(adv, dataset), dataset)
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
						best_adv = copy.deepcopy(adv)
						curr_pred = copy.deepcopy(torch.max(pred, 1)[1])

					# count = 0
					# for a in torch.eq(torch.max(pred, 1)[1], target).tolist():
					# 	count+=a
					# if count > prev_count:
					# 	best_adv = copy.deepcopy(adv)
					# if torch.all(torch.eq(torch.max(pred, 1)[1], target)):

					# 	break
				else:
					if best_adv != None:
						for i in range(len(best_adv)):
							if curr_pred[i]==target[i]:
								best_adv[i] = adv[i]
					else:
						best_adv = copy.deepcopy(adv)
						curr_pred = copy.deepcopy(torch.max(pred, 1)[1])


	return best_adv

def preprocess(img, dataset):
	"""
	Preprocess by normalization. 

	Single image only.
	"""
	if dataset=='multi-pie':
		img[:, 0] -= 131.0912
		img[:, 1] -= 103.8827
		img[:, 2] -= 91.4953
		img[:, 0] /= 1.0
		img[:, 1] /= 1.0
		img[:, 2] /= 1.0
	else:
		img[:, 0] -= 0.4914
		img[:, 1] -= 0.4822
		img[:, 2] -= 0.4465
		img[:, 0] /= 0.2023
		img[:, 1] /= 0.1994
		img[:, 2] /= 0.2010

	return img

def postprocess(img, dataset):
	"""
	Postprocess by unnormalizing

	Single image only.
	"""
	if dataset=='multi-pie':		
		img[:, 0] *= 1.0
		img[:, 1] *= 1.0
		img[:, 2] *= 1.0
		img[:, 0] += 131.0912
		img[:, 1] += 103.8827
		img[:, 2] += 91.4953
	else:
		img[:, 0] *= 0.2023
		img[:, 1] *= 0.1994
		img[:, 2] *= 0.2010
		img[:, 0] += 0.4914
		img[:, 1] += 0.4822
		img[:, 2] += 0.4465	

	return img

def clip_pixels(adv, dataset):
	"""
	Project img into 0-255 pixel range, based on norm. 

	Simply clamping inside the 0-1 range used in image representations of 
	tensors for the time being.
	"""
	# true_img = postprocess(img.clone().detach())
	# true_adv = torch.max(torch.min(adv, true_img+epsilon), true_img-epsilon)
	if dataset=='multi-pie':
		true_adv = adv.clamp(0, 255)
	else:
		true_adv = adv.clamp(0, 1)
	return true_adv

def get_inv_dwt_2d(LL, Y, ifm):
	"""
	Return idwt of bands.

	@ Attribute Y: The entire 3 stacked fine to coarse filter outputs
	"""
	return ifm((LL, Y))

def get_dwt_2d(img_batch, xfm):
	"""
	Return img dwt.
	
	Can probably improve runtimes by not creating a new DWT object per iter.
	"""
	LL, Y = xfm(img_batch)
	return LL, Y

def test_code(img):
	"""
	Ignore.
	"""
	# original = pywt.data.camera()

	titles = ['Original', 'Approximation', ' Horizontal detail',
	'Vertical detail', 'Diagonal detail', 'reconstructed']

	# print(original.shape)
	# coeffs2 = pywt.dwt2(img.data.cpu().squeeze().permute(1, 2, 0), 'bior1.3')
	# coeffs2 = pywt.dwt2(original, 'bior1.3')
	
	xfm = DWTForward(J=3, mode='zero', wave='db3')
	LL, Y = xfm(img.cpu())
	print(Y[0].size())
	LH, HL, HH = torch.unbind(Y[0], dim=2)
	Y = get_inv_dwt_2d(LL, Y)
	print('reconstructed img', Y.size())
	print(LH.shape)
	print(img.size())
	fig = plt.figure(figsize=(12, 3))
	for i, a in enumerate([img, LL, LH, HL, HH, Y]):
		ax = fig.add_subplot(1, 6, i + 1)
		try:
			ax.imshow(a.data.squeeze())
		except:
			ax.imshow(a.data.cpu().squeeze().permute(1, 2, 0))

		ax.set_title(titles[i], fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.plot

	fig.tight_layout()
	plt.savefig('./imgs/one.png')
