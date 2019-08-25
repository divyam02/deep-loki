import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.sampler as sampler
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from deepfool import *
from universal_pert import *
from utils.models import *


def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument('--net', dest='net',
						help='resnet50, densenet121',
						default='resnet50', type=str)
	parser.add_argument('--disp_interval', dest='disp_interval',
						default=10, type=int)
	parser.add_argument('--cuda', dest='cuda',	
						help='CUDA', action='store_true')
	parser.add_argument('--batch', dest='batch_size',
						help='batch size of training images', type=int,
						default=1000)
	parser.add_argument('--save_dir', dest='save_dir',
						help='save directory for perturbed images',
						default='./perturbed', type=str)
	parser.add_argument('--use_tfboard', dest='use_tfboard',
						help='use tf board?', default=False,
						type=bool)
	parser.add_argument('--dataset', dest='dataset',
						help='training images', type=str,
						default='cifar10')
	parser.add_argument('--seed', dest='seed',
						help='Add seed',
						type=int, default=22)
	parser.add_argument('--use_seed', dest='use_seed',
						type=bool, default=True)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()

	if args.dataset=='cifar10':	
		# Load all this from utils folder!
		morph = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
									std=[0.2023, 0.1994, 0.2010])])
		all_data = datasets.CIFAR10(root = '../data/', transform=morph, train=True, download=True)
		test_data = datasets.CIFAR10(root='../data/', transform=morph, train=False, download=True)


	elif args.dataset=='cifar100':
		all_data = datasets.CIFAR100(root = '../data/', train=True)
		test_data = datasets.CIFAR100(root='../data/', train=False)
	
	else:
		print("Unknown dataset. Help!")

	assert all_data is not None
	if args.use_seed:
		np.random.seed(args.seed)

	train_size = len(all_data)
	indices = list(range(train_size))
	np.random.shuffle(indices)
	split = int(np.floor(0.1*train_size))
	print(split)
	train_idx, val_idx = indices[split:], indices[:split]
	print(len(train_idx), len(val_idx))
	train_sampler = sampler.SubsetRandomSampler(train_idx)
	val_sampler = sampler.SubsetRandomSampler(val_idx)

	train_size = len(train_idx)
	val_size = len(val_idx)

	train_loader = torch.utils.data.DataLoader(all_data, batch_size=1,
												sampler=train_sampler)
	val_loader = torch.utils.data.DataLoader(all_data, batch_size=250,
												sampler=val_sampler)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=250,
											shuffle=False)

	if args.net=="resnet50":
		net = resnet50(pretrained=True) # CIFAR10 pretrained!

	elif args.net=="densenet121":
		net = densenet121(pretrained=True) # CIFAR10 pretrained!

	net.eval()

	if args.cuda:
		net.cuda()

	pert_file = '../utils/perturbations/universal_pert.npy'

	if os.path.isfile(pert_file):
		v = torch.from_numpy(np.load(pert_file)[0])
	else:
		v = get_univ_pert(train_loader, val_loader, net, args.cuda)
		np.save('pert_univ', v.detach().cpu().numpy())

	
	# A precaution. Not sure if net has been modified despite net.eval()
	if args.net=="resnet50":
		net = resnet50(pretrained=True) # CIFAR10 pretrained!

	elif args.net=="densenet121":
		net = densenet121(pretrained=True) # CIFAR10 pretrained!

	if args.cuda:
		net.cuda()

	net.eval()

	test_iter = iter(test_loader)
	print(len(test_loader.dataset))
	test_len = len(test_loader.dataset)//250
	correct = 0
	for i in range(test_len):
		with torch.no_grad():
			img, label = next(test_iter)
			img = img.cuda()
			output = net(img + v) # Use perturbed image!
			_, predicted = torch.max(output, 1)
			correct += (predicted==label).sum().item()
				

	print("Network accuracy on perturbed test data:", 100 * correct/total)

	"""
	train_iter = iter(train_loader)
	data = next(train_iter)
	print(data[0], data[1])
	print(all_data)
	"""

	# get training, validation and test sets (fixed)
		# What should be batchsize??