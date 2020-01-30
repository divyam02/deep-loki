import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_wavelets import DWTForward, DWTInverse
import pywt
import pywt.data
from torchvision import transforms as transforms
import torch.optim as optim

def perturb_img(img, label, target, model, alpha=0.5, max_iters=100):
	"""
	Assume learned filters are available.
	"""

	# Perform DWT on the unnormalized image. Ensure it is done on a copy.
	LL, Y = get_dwt_2d(postprocess(img.clone().detach()))

	# Enable updating of LL, LH, HL, HH.
	LL = LL.clone().requires_grad_().cuda()
	Y[0] = Y[0].clone().requires_grad_().cuda()
	Y[1] = Y[1].clone().requires_grad_().cuda()
	Y[2] = Y[2].clone().requires_grad_().cuda()

	optimizer = optim.Adam(model.parameters(), lr=1e-4)

	cls_loss_fn = torch.nn.CrossEntropyLoss()
	dst_loss_fn = torch.nn.PairwiseDistance()

	for step in range(max_iters):
		# Obtain unnormalized adversarial image by 
		# using IDWT on LL, LH, HL, HH.
		adv = get_inv_dwt_2d(LL, Y)

		# Preprocess image for the network.
		adv = preprocess(clip_pixels(adv))
		optimizer.zero_grad()
		print('step', step+1)
		pred = model(adv)

		# Control loss tradeoffs with alpha.
		loss = (1-alpha)*cls_loss_fn(pred, target) + (alpha)*dst_loss_fn(torch.flatten(adv, 1, -1), 
															torch.flatten(img, 1, -1))**2
		loss.backward()
		optimizer.step()

		# Check adversary after updating image.
		with torch.no_grad():
			adv = get_inv_dwt_2d(LL, Y)
			adv = preprocess(clip_pixels(adv))
			pred = model(adv)
			if torch.max(pred, 1)[1]==target:
				print('Success!')
				break

	return adv

def preprocess(img):
	"""
	Preprocess by normalization. 

	Single image only.
	"""
	img[0][0] -= 0.4914
	img[0][1] -= 0.4822
	img[0][2] -= 0.4465
	img[0][0] /= 0.2023
	img[0][1] /= 0.1994
	img[0][2] /= 0.2010

	return img

def postprocess(img):
	"""
	Postprocess by unnormalizing

	Single image only.
	"""
	img[0][0] *= 0.2023
	img[0][1] *= 0.1994
	img[0][2] *= 0.2010
	img[0][0] += 0.4914
	img[0][1] += 0.4822
	img[0][2] += 0.4465	

	return img

def clip_pixels(adv):
	"""
	Project img into 0-255 pixel range, based on norm. 

	Simply clamping inside the 0-1 range used in image representations of 
	tensors for the time being.
	"""
	# true_img = postprocess(img.clone().detach())
	# true_adv = torch.max(torch.min(adv, true_img+epsilon), true_img-epsilon)
	true_adv = adv.clamp(0, 1)
	return true_adv

def get_inv_dwt_2d(LL, Y):
	"""
	Return idwt of bands.

	@ Attribute Y: The entire 3 stacked fine to coarse filter outputs
	"""
	ifm = DWTInverse(mode='zero', wave='db3').cuda().eval()
	return ifm((LL, Y))

def get_dwt_2d(img_batch):
	"""
	Return img dwt.
	
	Can probably reduce memory load by not creating a new DWT object per iter.
	"""
	xfm = DWTForward(J=3, mode='zero', wave='db3').cuda().eval()
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

