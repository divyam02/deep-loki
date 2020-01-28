import numpy as np
import matplotlib.pyplot as plt
import torch
import pywt
import pywt.data


def perturb_img(img):

	# original = pywt.data.camera()

	titles = ['Approximation', ' Horizontal detail',
	'Vertical detail', 'Diagonal detail']

	print(img.size())
	coeffs2 = pywt.dwt2(img.data.cpu().squeeze().permute(1, 2, 0), 'bior1.3')
	LL, (LH, HL, HH) = coeffs2
	print(LL.shape)
	fig = plt.figure(figsize=(12, 4))
	for i, a in enumerate([LL, LH, HL, HH]):
		ax = fig.add_subplot(1, 4, i + 1)
		ax.imshow(torch.tensor(a).permute(1, 2, 0).numpy())
		ax.set_title(titles[i], fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.plot

	fig.tight_layout()
	plt.savefig('./imgs/one.png')
