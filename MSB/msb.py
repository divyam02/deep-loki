import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def add_msb_noise(net, img, fraction=0.1, to_numpy=False):
	"""
	Select three sets of pixels randomly across all channels.
	Not sure about how to control perturbation size...
	Shouldn't be much, this is only a 3x32x32 pixel image...

	Cap L2 norm of distance for now.
	"""

	og_img = torch.clone(img)

	og_img[0][0] *= 0.2023
	og_img[0][1] *= 0.1994
	og_img[0][2] *= 0.2010
	og_img[0][0] += 0.4914
	og_img[0][1] += 0.4822
	og_img[0][2] += 0.4465

	og_img *= 255

	og_img.clamp(0, 255)
	og_img = torch.tensor(og_img, dtype=torch.uint8)

	# print(og_img)

	perts = int(3*32*32*fraction*3)
	prev = dict()
	counter = 0
	turn = 0
	while True:
		i = random.randint(0, 2)
		j = random.randint(0, 31)
		k = random.randint(0, 31)

		turn = turn%3

		if (turn, i, j, k) in prev.keys():
			continue
		prev[(turn, i, j, k)] = None

		if turn==0:
			og_img[0, i, j, k] = og_img[0, i, j, k] ^ 128
		elif turn==1:
			og_img[0, i, j, k] = og_img[0, i, j, k] ^ 64
		else:
			og_img[0, i, j, k] = og_img[0, i, j, k] ^ 32

		if counter==perts:
			break

		counter+=1
		turn+=1

	og_img = og_img.float()
	og_img /= 255
	og_img[0][0] -= 0.4914
	og_img[0][1] -= 0.4822
	og_img[0][2] -= 0.4465
	og_img[0][0] /= 0.2023
	og_img[0][1] /= 0.1994
	og_img[0][2] /= 0.2010

	# print(og_img)
	# input("continue?")

	return og_img.cuda()

