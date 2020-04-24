import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision import transforms as transforms
import torch.nn.functional as F


def mix_bands(img_1, img_2, net, label_1, label_2, dataset, switch_band):
	"""
	Parameters:
		img_1: 1x3xHxW RGB image
		img_2: 1x3xHxW RGB image
	"""

	img_1 = torch.unsqueeze(img_1.clone(), 0)
	img_2 = torch.unsqueeze(img_2.clone(), 0)

	img_1 = postprocess(img_1, dataset)
	img_2 = postprocess(img_2, dataset)

	img_1_og = img_1.clone()
	img_2_og = img_2.clone()

	ifm = DWTInverse(mode='zero', wave='haar').cuda()
	xfm = DWTForward(J=1, mode='zero', wave='haar').cuda()

	fig, ax = plt.subplots(nrows=2, ncols=10, figsize=(20, 10))

	for i, img in enumerate([img_1, img_2]):
		LL, Y = xfm(img)
		LH, HL, HH = torch.unbind(Y[0], dim=2)

		ax[i, 0].imshow(LL[0].data.cpu().permute(1, 2, 0))
		ax[i, 1].imshow(10*LH[0].data.cpu().permute(1, 2, 0)/torch.max(LH[0].data.cpu().permute(1, 2, 0)))
		ax[i, 2].imshow(10*HL[0].data.cpu().permute(1, 2, 0)/torch.max(HL[0].data.cpu().permute(1, 2, 0)))
		ax[i, 3].imshow(10*HH[0].data.cpu().permute(1, 2, 0)/torch.max(HH[0].data.cpu().permute(1, 2, 0)))
		ax[i, 4].imshow(img[0].data.cpu().permute(1, 2, 0))

		ax[i, 0].set_title('LL'+'_'+str(i))
		ax[i, 1].set_title('LH'+'_'+str(i))
		ax[i, 2].set_title('HL'+'_'+str(i))
		ax[i, 3].set_title('HH'+'_'+str(i))
		ax[i, 4].set_title('normal_img'+'_'+str(i))

		ax[i, 0].set_yticks([], [])
		ax[i, 0].set_xticks([], [])
		ax[i, 1].set_yticks([], [])
		ax[i, 1].set_xticks([], [])
		ax[i, 2].set_yticks([], [])
		ax[i, 2].set_xticks([], [])
		ax[i, 3].set_yticks([], [])
		ax[i, 3].set_xticks([], [])
		ax[i, 4].set_yticks([], [])
		ax[i, 4].set_xticks([], [])

	# Reconstruct using new components
	LL_1, Y_1 = xfm(img_1)
	LH_1, HL_1, HH_1 = torch.unbind(Y_1[0], dim=2)

	LL_2, Y_2 = xfm(img_2)
	LH_2, HL_2, HH_2 = torch.unbind(Y_2[0], dim=2)

	if switch_band=='ll':
		img_1 = ifm((LL_2, [torch.stack((LH_1, HL_1, HH_1), 2)]))
		img_2 = ifm((LL_1, [torch.stack((LH_2, HL_2, HH_2), 2)]))

	elif switch_band=='lh':
		img_1 = ifm((LL_1, [torch.stack((LH_2, HL_1, HH_1), 2)]))
		img_2 = ifm((LL_2, [torch.stack((LH_1, HL_2, HH_2), 2)]))

	elif switch_band=='hl':
		img_1 = ifm((LL_1, [torch.stack((LH_1, HL_2, HH_1), 2)]))
		img_2 = ifm((LL_2, [torch.stack((LH_2, HL_1, HH_2), 2)]))

	elif switch_band=='hh':
		img_1 = ifm((LL_1, [torch.stack((LH_1, HL_1, HH_2), 2)]))
		img_2 = ifm((LL_2, [torch.stack((LH_2, HL_2, HH_1), 2)]))

	elif switch_band=='high':
		img_1 = ifm((LL_1, [torch.stack((LH_2, HL_2, HH_2), 2)]))
		img_2 = ifm((LL_2, [torch.stack((LH_1, HL_1, HH_1), 2)]))


	epsilon = 0.031
	eta = torch.clamp(img_1.data - img_1_og, -epsilon, epsilon)
	img_1 = img_1_og + eta
	img_1 = torch.clamp(img_1, 0, 1.0)
	
	eta = torch.clamp(img_2.data - img_2_og, -epsilon, epsilon)
	img_2 = img_2_og + eta
	img_2 = torch.clamp(img_2, 0, 1.0)

	for i, img in enumerate([img_1, img_2]):
		LL, Y = xfm(img)
		LH, HL, HH = torch.unbind(Y[0], dim=2)

		ax[i, 0+5].imshow(LL[0].data.cpu().permute(1, 2, 0))
		ax[i, 1+5].imshow(10*LH[0].data.cpu().permute(1, 2, 0)/torch.max(LH[0].data.cpu().permute(1, 2, 0)))
		ax[i, 2+5].imshow(10*HL[0].data.cpu().permute(1, 2, 0)/torch.max(HL[0].data.cpu().permute(1, 2, 0)))
		ax[i, 3+5].imshow(10*HH[0].data.cpu().permute(1, 2, 0)/torch.max(HH[0].data.cpu().permute(1, 2, 0)))
		ax[i, 4+5].imshow(img[0].data.cpu().permute(1, 2, 0))

		ax[i, 0+5].set_title('LL'+'_'+str(i))
		ax[i, 1+5].set_title('LH'+'_'+str(i))
		ax[i, 2+5].set_title('HL'+'_'+str(i))
		ax[i, 3+5].set_title('HH'+'_'+str(i))
		ax[i, 4+5].set_title('perturbed_img'+'_'+str(i))

		ax[i, 0+5].set_yticks([], [])
		ax[i, 0+5].set_xticks([], [])
		ax[i, 1+5].set_yticks([], [])
		ax[i, 1+5].set_xticks([], [])
		ax[i, 2+5].set_yticks([], [])
		ax[i, 2+5].set_xticks([], [])
		ax[i, 3+5].set_yticks([], [])
		ax[i, 3+5].set_xticks([], [])
		ax[i, 4+5].set_yticks([], [])
		ax[i, 4+5].set_xticks([], [])

	plt.savefig('./imgs/mix.png')

	print(['*']*20)

	resize = transforms.Resize((32, 32))

	LL_1, Y_1 = xfm(preprocess(img_1, dataset))
	LH_2, HL_2, HH_2 = torch.unbind(Y_1[0], dim=2)

	LL_2, Y_2 = xfm(preprocess(img_2, dataset))
	LH_1, HL_1, HH_1 = torch.unbind(Y_2[0], dim=2)

	print('image 1 label:\t', label_1)
	print('perturbed label:\t', torch.max(net(preprocess(img_1, dataset)), 1)[1])
	print('print LH_1 label:\t', torch.max(net(F.interpolate(LH_1, 32)), 1)[1])
	print('print HL_1 label:\t', torch.max(net(F.interpolate(HL_1, 32)), 1)[1])
	print('print HH_1 label:\t', torch.max(net(F.interpolate(HH_1, 32)), 1)[1])
	print('image 2 label:\t', label_2)
	print('perturbed label:\t', torch.max(net(preprocess(img_2, dataset)), 1)[1])
	print('print LH_2 label:\t', torch.max(net(F.interpolate(LH_2, 32)), 1)[1])
	print('print HL_2 label:\t', torch.max(net(F.interpolate(HL_2, 32)), 1)[1])
	print('print HH_2 label:\t', torch.max(net(F.interpolate(HH_2, 32)), 1)[1])


def preprocess(img, dataset=None, net_type=None):
	"""
	Preprocess by normalization. 

	Single image only.
	"""
	temp = img.clone()

	if net_type=='inception':
		temp = (temp - 127.5)/128.0
	
	elif dataset=='multi-pie':
		temp[:, 0] -= 131.0912
		temp[:, 1] -= 103.8827
		temp[:, 2] -= 91.4953
		temp[:, 0] /= 1.0
		temp[:, 1] /= 1.0
		temp[:, 2] /= 1.0

	elif dataset=='fmnist':
		return img

	elif dataset=="tinyimagenet":
		temp[:, 0] -= 0.4802
		temp[:, 1] -= 0.4481
		temp[:, 2] -= 0.3975
		temp[:, 0] /= 0.2302
		temp[:, 1] /= 0.2265
		temp[:, 2] /= 0.2262
	elif dataset=="cifar10-trades":
		pass
	elif dataset=='mnist':
		pass
	elif dataset=="cifar10-feat-scat":
		temp[:, 0] -= 0.5
		temp[:, 1] -= 0.5
		temp[:, 2] -= 0.5
		temp[:, 0] /= 0.5
		temp[:, 1] /= 0.5
		temp[:, 2] /= 0.5
	elif dataset=="imagenet":
		temp[:, 0] -= 0.485
		temp[:, 1] -= 0.456
		temp[:, 2] -= 0.406
		temp[:, 0] /= 0.229
		temp[:, 1] /= 0.224
		temp[:, 2] /= 0.225
	else:
		temp[:, 0] -= 0.4914
		temp[:, 1] -= 0.4822
		temp[:, 2] -= 0.4465
		temp[:, 0] /= 0.2023
		temp[:, 1] /= 0.1994
		temp[:, 2] /= 0.2010

	return temp

def postprocess(img, dataset=None, net_type=None):
	"""
	Postprocess by unnormalizing

	Single image only.
	"""
	temp = img.clone()

	if net_type=='inception':
		img = (img * 128.0) + 127.5
	
	elif dataset=='multi-pie':		
		temp[:, 0] *= 1.0
		temp[:, 1] *= 1.0
		temp[:, 2] *= 1.0
		temp[:, 0] += 131.0912
		temp[:, 1] += 103.8827
		temp[:, 2] += 91.4953

	elif dataset=='fmnist':
		return img
	elif dataset=='tinyimagenet':
		temp[:, 0] *= 0.2302
		temp[:, 1] *= 0.2265
		temp[:, 2] *= 0.2262
		temp[:, 0] += 0.4802
		temp[:, 1] += 0.4481
		temp[:, 2] += 0.3975
	elif dataset=="cifar10-trades":
		pass
	elif dataset=='mnist':
		pass
	elif dataset=="cifar10-feat-scat":
		temp[:, 0] *= 0.5
		temp[:, 1] *= 0.5
		temp[:, 2] *= 0.5
		temp[:, 0] += 0.5
		temp[:, 1] += 0.5
		temp[:, 2] += 0.5
	elif dataset=="imagenet":
		temp[:, 0] *= 0.229
		temp[:, 1] *= 0.224
		temp[:, 2] *= 0.225	
		temp[:, 0] += 0.485
		temp[:, 1] += 0.456
		temp[:, 2] += 0.406
	else:
		temp[:, 0] *= 0.2023
		temp[:, 1] *= 0.1994
		temp[:, 2] *= 0.2010
		temp[:, 0] += 0.4914
		temp[:, 1] += 0.4822
		temp[:, 2] += 0.4465	

	return temp
