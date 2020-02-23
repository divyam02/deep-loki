import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.sampler as sampler
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from batch_dwt_attack import *
import sys
from utils.models import *
import cv2

import logging

def side_plot(og_img, pert_img, label, k_i, j, dataset):
	"""
	Quick visual debug!
	"""
	classes_multi_pie = range(0, 377)
	classes_tinyimagenet = range(0,200)
	classes_cifar_10 = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# classes = classes_cifar_10
	# classes = classes_tinyimagenet
	classes = classes_multi_pie

	np.save('./npy_files/adv_'+str(label), pert_img.data.cpu().numpy())

	if dataset=='multi-pie':
		og_img[:, 0] *= 1.0
		og_img[:, 1] *= 1.0
		og_img[:, 2] *= 1.0
		og_img[:, 0] += 131.0912
		og_img[:, 1] += 103.8827
		og_img[:, 2] += 91.4953

		pert_img[:, 0] *= 1.0
		pert_img[:, 1] *= 1.0
		pert_img[:, 2] *= 1.0
		pert_img[:, 0] += 131.0912
		pert_img[:, 1] += 103.8827
		pert_img[:, 2] += 91.4953	

	else:
		og_img[:, 0] *= 0.2023
		og_img[:, 1] *= 0.1994
		og_img[:, 2] *= 0.2010
		og_img[:, 0] += 0.4914
		og_img[:, 1] += 0.4822
		og_img[:, 2] += 0.4465

		pert_img[:, 0] *= 0.2023
		pert_img[:, 1] *= 0.1994
		pert_img[:, 2] *= 0.2010
		pert_img[:, 0] += 0.4914
		pert_img[:, 1] += 0.4822
		pert_img[:, 2] += 0.4465

	_, ax = plt.subplots(nrows=list(og_img.size())[0], ncols=4, figsize=(15, 60))

	for i in range(list(og_img.size())[0]):

		logger = logging.getLogger() # For removing clipping message
		old_level = logger.level
		logger.setLevel(100)

		ax[i, 0].imshow(og_img[i].data.cpu().permute(1, 2, 0))
		ax[i, 1].imshow(pert_img[i].data.cpu().permute(1, 2, 0))

		# Pixel differences are of the order of 1e-7. Current method to visualize these
		# values. I will fix this soon.
		# if dataset=='multi-pie':
		# 	ax[i, 2].imshow((torch.abs(og_img[i] - pert_img[i])).data.cpu().permute(1, 2, 0))
		# 	ax[i, 3].imshow((torch.abs(og_img[i] - pert_img[i])).data.cpu().permute(1, 2, 0))			
		# else:
		ax[i, 2].imshow((torch.abs(og_img[i] - pert_img[i])*10).data.cpu().permute(1, 2, 0))
		ax[i, 3].imshow((torch.abs(og_img[i] - pert_img[i])*100).data.cpu().permute(1, 2, 0))

		ax[i, 0].set_title(classes[label[i]])
		ax[i, 0].set_yticks([], [])
		ax[i, 0].set_xticks([], [])

		ax[i, 1].set_title(classes[k_i[i]])
		ax[i, 1].set_yticks([], [])
		ax[i, 1].set_xticks([], [])

		ax[i, 2].set_title('10xNoise')
		ax[i, 2].set_yticks([], [])
		ax[i, 2].set_xticks([], [])

		ax[i, 3].set_title('100xNoise')
		ax[i, 3].set_yticks([], [])
		ax[i, 3].set_xticks([], [])

		logger.setLevel(old_level) # Clipping message removal

		# A second confirmation about the presence of noise.
		# print('Noise is present?', torch.all(torch.abs(og_img - pert_img)==0))

	plt.savefig('./imgs/works_'+str(j)+'.png')
	plt.close()

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument('--net', dest='net',
						help='resnet50, densenet121',
						default='resnet50-cifar', type=str)
	parser.add_argument('--disp_interval', dest='disp_interval',
						default=10, type=int)
	parser.add_argument('--cuda', dest='cuda',	
						help='CUDA', action='store_true')
	parser.add_argument('--batch', dest='batch_size',
						help='batch size of training images', type=int,
						default=25)
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
	parser.add_argument('--attack_type', dest='attack_type',
						type=str, default='max_error')
	parser.add_argument('--random_restarts', dest='random_restarts',
						type=bool, default=False)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()

	if args.dataset=='cifar10':	
		# Load all this from utils folder!
		morph = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
									std=[0.2023, 0.1994, 0.2010])])
		test_data = datasets.CIFAR10(root='../data/', transform=morph, train=False, download=True)


	elif args.dataset=='cifar100':
		test_data = datasets.CIFAR100(root='../data/', train=False)

	elif args.dataset=='tinyimagenet':

		path = '../data/tiny-imagenet-200/val/images' # path where validation data is present now
		filename = '../data/tiny-imagenet-200/val/val_annotations.txt'  # file where image2class mapping is present
		fp = open(filename, "r")  # open file in read mode
		data = fp.readlines()  # read line by line

		# Create a dictionary with image names as key and corresponding classes as values
		val_img_dict = {}
		for line in data:
		    words = line.split("\t")
		    val_img_dict[words[0]] = words[1]
		fp.close()

		# Create folder if not present, and move image into proper folder
		for img, folder in val_img_dict.items():
			newpath = (os.path.join(path, folder))
			if not os.path.exists(newpath):  # check if folder exists
				os.makedirs(newpath)

			if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
				os.rename(os.path.join(path, img), os.path.join(newpath, img))


		morph = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
							std=[0.2023, 0.1994, 0.2010])])

		test_data = datasets.ImageFolder(root='../data/tiny-imagenet-200/val/images', transform=morph)


	elif args.dataset=='multi-pie':

		path = '../../mpie/val'
		morph = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), 
				transforms.Normalize(mean=[131.0912, 103.8827, 91.4953], std=[1, 1, 1])])

		test_data = datasets.ImageFolder(root=path, transform=morph)
		
	else:
		print("Unknown dataset. Help!")

	if args.use_seed:
		np.random.seed(args.seed)

	# Convert to batches for faster testing.
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
											shuffle=True)

	if args.net=="resnet50-cifar":
		print("Using CIFAR-10 pretrained ResNet-50")
		net = resnet50(pretrained=True) # CIFAR10 pretrained!

	elif args.net=="densenet121":
		print("Using CIFAR-10 pretrained DenseNet-121")
		net = densenet121(pretrained=True) # CIFAR10 pretrained!

	elif args.net=="resnet50-imagenet":
		print('Using ImageNet pretrained ResNet-50')
		net = torchvision.models.resnet50(pretrained=True)

	elif args.net=="resnet50-madry":
		print("Using Madry pretrained ResNet-50")

	elif args.net=="vggface2":
		print("Using Oxford VGG Net")
		from resnet50_scratch_dims_2048 import *
		net = resnet50_scratch(weights_path='./resnet50_scratch_dims_2048.pth')
		net.classifier = nn.Conv2d(2048, 337, kernel_size=[1, 1], stride=(1, 1))
		net.load_state_dict(torch.load('../../drive/My Drive/multi_pie_data/best_model.pth')['model_state_dict'])

	print("Freezing network layers and setting to eval mode")
	net.eval()
	for params in net.parameters():
		params.requires_grad = False

	if args.cuda:
		net.cuda()

	test_iter = iter(test_loader)
	print(len(test_loader.dataset))
	test_len = len(test_loader.dataset)
	total = len(test_loader.dataset)

	miss_correct = 0 # Misclassifies as target.
	miss_incorrect = 0 # Misclassifies, but not as target.
	correct = 0 # Classifies correctly.

	for i in range(test_len):

		img, label = next(test_iter)
		img, label = img.cuda(), label.cuda()

		# Arbitrarily kept 'truck' label as the target class.
		# All 'truck' catergory images are to misclassified as
		# the 'plane' catergory (arbitrary choice). 
		if args.attack_type=='next_class':
			if args.dataset=='cifar10':target = (label.clone().detach() + 1)%10
			if args.dataset=='tinyimagenet':target = (label.clone().detach() + 1)%200
			if args.dataset=='multi-pie':target = (label.clone().detach() + 1)%337
		if args.attack_type=='max_error':
			if args.dataset=='cifar10':target = label.clone().detach()
			if args.dataset=='tinyimagenet':target = label.clone().detach()
			if args.dataset=='multi-pie':target = label.clone().detach()

		pert_img = perturb_img(img.clone().detach(), label, target, net, args.attack_type, 
								args.dataset, random_restarts=args.random_restarts)

		output = net(pert_img)
		_, predicted = torch.max(output, 1)
		if i%10==0:
			print("Label:", label.data, "Predicted:", predicted.data, "Iter:", i)
		if not torch.all(torch.eq(predicted, target)): 
			print('\nAttack Failed! Image batch:', i)
			print('Label:', label)
			print('Predicted:', predicted)

		miss_correct+=(predicted==target).sum().item()
		for k, j in zip(torch.eq(predicted, target).tolist(), torch.eq(predicted, label).tolist()):
			miss_incorrect+=(not(k or j))
		correct+=(predicted==label).sum().item()
		if (i+1)*args.batch_size%25==0:
			side_plot(img, pert_img, label, predicted, i, args.dataset)
			print("\nMisclassified as target:\t\t", miss_correct/((i+1)*args.batch_size))
			print("Misclassified, but not as target:\t", miss_incorrect/((i+1)*args.batch_size))
			print("Network accuracy on perturbed test data:", correct/((i+1)*args.batch_size))
			print("Processed:", (i+1)*args.batch_size)

	print("Network accuracy on perturbed test data:", correct/total)