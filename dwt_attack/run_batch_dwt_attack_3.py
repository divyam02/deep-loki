import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data.sampler as sampler
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import sys
from utils.models import *
import cv2
from scipy.ndimage import gaussian_filter
from mixup_bands import *
from collections import OrderedDict
from kernel_def import *
from batch_dwt_attack_3 import *
from uiqi import *

from scipy.io import savemat

import logging

"""

Utility functions

"""

def gauss_side_plot(og_img, pert_img, gauss_img, dataset, j, net_type):
	classes_multi_pie = range(0, 377)
	classes_tinyimagenet = range(0,200)
	classes_cifar_10 = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	classes_fmnist = ('top', 'trouser', 'pullover', 'dress', 'coat', 
			'sandal', 'shirt', 'sneaker', 'bag', 'boot')
	# classes = classes_cifar_10
	# classes = classes_tinyimagenet
	classes = classes_multi_pie
	# classes = classes_fmnist

	# np.save('./npy_files/adv_'+str(label), pert_img.data.cpu().numpy())

	og_img = postprocess(og_img, dataset, net_type)
	gauss_img = postprocess(gauss_img, dataset, net_type)
	pert_img = postprocess(pert_img, dataset, net_type)

	_, ax = plt.subplots(nrows=list(og_img.size())[0], ncols=3, figsize=(15, 60))

	for i in range(list(og_img.size())[0]):

		logger = logging.getLogger() # For removing clipping message
		old_level = logger.level
		logger.setLevel(100)

		if dataset=='fmnist':
			ax[i, 0].imshow(og_img[i,0].data.cpu())
			ax[i, 1].imshow(pert_img[i,0].data.cpu())	
			ax[i, 2].imshow(gauss_img[i,0].data.cpu())
		else:
			ax[i, 0].imshow(og_img[i].data.cpu().permute(1, 2, 0))
			ax[i, 1].imshow(pert_img[i].data.cpu().permute(1, 2, 0))
			ax[i, 2].imshow(gauss_img[i].data.cpu().permute(1, 2, 0))

		# ax[i, 2].imshow((torch.abs(og_img[i] - pert_img[i])*10).data.cpu().permute(1, 2, 0))
		# ax[i, 3].imshow((torch.abs(og_img[i] - pert_img[i])*100).data.cpu().permute(1, 2, 0))

		ax[i, 0].set_title('original image')
		ax[i, 0].set_yticks([], [])
		ax[i, 0].set_xticks([], [])

		ax[i, 1].set_title('perturbed image')
		ax[i, 1].set_yticks([], [])
		ax[i, 1].set_xticks([], [])

		ax[i, 2].set_title('gaussian smoothing')
		ax[i, 2].set_yticks([], [])
		ax[i, 2].set_xticks([], [])		

		logger.setLevel(old_level) # Clipping message removal

		# A second confirmation about the presence of noise.
		# print('Noise is present?', torch.all(torch.abs(og_img - pert_img)==0))

	plt.savefig('./imgs/gauss_works_'+str(j)+'.png')
	plt.close()

def side_plot(og_img, pert_img, label, k_i, j, dataset, net_type):
	"""
	Quick visual debug!
	"""
	classes_multi_pie = range(0, 377)
	classes_tinyimagenet = range(0,200)
	classes_cifar_10 = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	classes_fmnist = ('top', 'trouser', 'pullover', 'dress', 'coat', 
			'sandal', 'shirt', 'sneaker', 'bag', 'boot')
	if dataset=='fmnist':
		classes = classes_fmnist
	elif dataset=='multi-pie':
		classes = classes_multi_pie
	elif dataset=='tiny-imagenet':
		classes = classes_tinyimagenet
	else:
		classes = classes_cifar_10


	# np.save('./npy_files/adv_'+str(label), pert_img.data.cpu().numpy())

	og_img = postprocess(og_img, dataset, net_type)
	pert_img = postprocess(pert_img, dataset, net_type)

	if dataset=='multi-pie' and net_type=='vggface2':
		og_img/=255.0
		pert_img/=255.0

	_, ax = plt.subplots(nrows=list(og_img.size())[0], ncols=4, figsize=(10, 60))

	for i in range(list(og_img.size())[0]):

		logger = logging.getLogger() # For removing clipping message
		old_level = logger.level
		logger.setLevel(100)

		if dataset=='fmnist':
			ax[i, 0].imshow(og_img[i,0].data.cpu())
			ax[i, 1].imshow(pert_img[i,0].data.cpu())	
			ax[i, 2].imshow((torch.abs(og_img[i,0] - pert_img[i,0])*20).data.cpu())
			ax[i, 3].imshow((torch.abs(og_img[i,0] - pert_img[i,0])*100).data.cpu())		
		else:
			ax[i, 0].imshow(og_img[i].data.cpu().permute(1, 2, 0))
			ax[i, 1].imshow(pert_img[i].data.cpu().permute(1, 2, 0))
			ax[i, 2].imshow((torch.abs(og_img[i] - pert_img[i])*20).data.cpu().permute(1, 2, 0))
			ax[i, 3].imshow((torch.abs(og_img[i] - pert_img[i])*100).data.cpu().permute(1, 2, 0))

		# ax[i, 2].imshow((torch.abs(og_img[i] - pert_img[i])*10).data.cpu().permute(1, 2, 0))
		# ax[i, 3].imshow((torch.abs(og_img[i] - pert_img[i])*100).data.cpu().permute(1, 2, 0))

		# ax[i, 0].set_title(str(classes[label[i]])+':\n'+str(clean_conf[i].data))
		ax[i, 0].set_yticks([], [])
		ax[i, 0].set_xticks([], [])

		# ax[i, 1].set_title(str(classes[k_i[i]])+':\n'+str(pert_conf[i].data))
		ax[i, 1].set_yticks([], [])
		ax[i, 1].set_xticks([], [])

		# ax[i, 2].set_title('20xNoise')
		ax[i, 2].set_yticks([], [])
		ax[i, 2].set_xticks([], [])

		# ax[i, 3].set_title('100xNoise')
		ax[i, 3].set_yticks([], [])
		ax[i, 3].set_xticks([], [])

		logger.setLevel(old_level) # Clipping message removal

		# A second confirmation about the presence of noise.
		# print('Noise is present?', torch.all(torch.abs(og_img - pert_img)==0))

	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	plt.savefig('./imgs/works_'+dataset+'.png')
	plt.close()

def save_filters(img_1, img_2, dataset, net_type):
	img_1 = torch.unsqueeze(img_1.clone(), 0)
	img_2 = torch.unsqueeze(img_2.clone(), 0)

	img_1 = postprocess(img_1, dataset, net_type)
	img_2 = postprocess(img_2, dataset, net_type)

	if dataset=='multi-pie' and net_type=='vggface2':
		img_1/=255.0
		img_2/=255.0

	img_1_og = img_1.clone()
	img_2_og = img_2.clone()

	ifm = DWTInverse(mode='zero', wave='haar').cuda()
	xfm = DWTForward(J=1, mode='zero', wave='haar').cuda()
	
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(9, 4))
	
	title = ['clean', 'perturbed']

	for i, img in enumerate([img_1, img_2]):
		LL, Y = xfm(img)
		LH, HL, HH = torch.unbind(Y[0], dim=2)

		ax[i, 1].imshow(LL[0].data.cpu().permute(1, 2, 0))
		ax[i, 2].imshow(10*LH[0].data.cpu().permute(1, 2, 0)/torch.max(LH[0].data.cpu().permute(1, 2, 0)))
		ax[i, 3].imshow(10*HL[0].data.cpu().permute(1, 2, 0)/torch.max(HL[0].data.cpu().permute(1, 2, 0)))
		ax[i, 4].imshow(10*HH[0].data.cpu().permute(1, 2, 0)/torch.max(HH[0].data.cpu().permute(1, 2, 0)))
		ax[i, 0].imshow(img[0].data.cpu().permute(1, 2, 0))

		ax[i, 1].set_title(title[i]+' LL')
		ax[i, 2].set_title(title[i]+' LH')
		ax[i, 3].set_title(title[i]+' HL')
		ax[i, 4].set_title(title[i]+' HH')
		ax[i, 0].set_title(title[i]+' image')

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

	fig.tight_layout()
	plt.savefig('./imgs/bands.png')
	# print(pert_img.size())
	# input('continue')

"""

Main section of the code

"""
def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument('--net', dest='net',help='resnet50, densenet121',
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
	parser.add_argument('--gaussian_smoothing', dest='gaussian_smoothing',
						type=bool, default=False)
	parser.add_argument('--sigma', dest='sigma',
						type=float, default=1.0)
	parser.add_argument('--special_test', dest='special_test',
						type=bool, default=False)
	parser.add_argument('--defence_type', default='',
						type=str, dest='defence_type')
	parser.add_argument('--bands', dest='bands',
						type=str, default=None)
	parser.add_argument('--update_type', dest='update_type',
						type=str, default='gradient')
	parser.add_argument('--odi', dest='odi',
						type=bool, default=False)
	parser.add_argument('--loss_type', dest='loss_type',
						type=str, default='xent')
	parser.add_argument('--rho', dest='rho',
						type=float, default=1.0)
	parser.add_argument('--shuffle', dest='shuffle',
						type=bool, default=False)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	"""
	
	The dataset is preprocessed and converted into a suitable object for the dataloader.
	Key arguments:
		1. morph: Sequence of preprocessing methods.
		2. test_data: init for DataLoader class object test_loader.

	"""
	args = parse_args()

	if args.dataset=='cifar10':	
		# Load all this from utils folder!
		morph = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
									std=[0.2023, 0.1994, 0.2010])])
		test_data = datasets.CIFAR10(root='../data/', transform=morph, train=False, download=True)

	elif args.dataset=='imagenet':	
		# Load all this from utils folder!
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
		morph = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), 
                                transforms.ToTensor(), normalize])
		test_data = torchvision.datasets.ImageFolder(root='../../drive/My Drive/imagenet_validation', transform=morph)

	elif args.dataset=='imagenet-test':	
		# Load all this from utils folder!
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
		morph = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), 
                                transforms.ToTensor(), normalize])
		test_data = torchvision.datasets.ImageFolder(root='../../drive/My Drive/imagenet_test', transform=morph)
		args.dataset = 'imagenet'

	elif args.dataset=='cifar10-trades':	
		# Load all this from utils folder!
		morph = transforms.Compose([transforms.ToTensor()])
		test_data = datasets.CIFAR10(root='../data/', transform=morph, train=False, download=True)

	elif args.dataset=='cifar10-feat-scat':	
		# Load all this from utils folder!
		morph = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], 
									std=[0.5, 0.5, 0.5])])
		test_data = datasets.CIFAR10(root='../data/', transform=morph, train=False, download=True)

	elif args.dataset=='cifar100':
		test_data = datasets.CIFAR100(root='../data/', train=False)

	elif args.dataset=='tiny-imagenet':
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

		morph = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], 
							std=[0.2302, 0.2265, 0.2262])])

		test_data = datasets.ImageFolder(root='../data/tiny-imagenet-200/val/images', transform=morph)

	elif args.dataset=='multi-pie' and args.net=='inception':
		from facenet_pytorch import fixed_image_standardization
		path = '../../mpie/test'
		morph = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor(), fixed_image_standardization])
		test_data = datasets.ImageFolder(root=path, transform=morph)

	elif args.dataset=='multi-pie':
		path = '../../mpie/test'
		morph = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0,0,0],
				std=[1/255.0, 1/255.0, 1/255.0]), transforms.Normalize(mean=[131.0912, 103.8827, 91.4953], std=[1, 1, 1])])
		test_data = datasets.ImageFolder(root=path, transform=morph)
		
	elif args.dataset=='mnist':
		transform_test = transforms.Compose([transforms.ToTensor(),])
		test_data = datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)

	elif args.dataset=='fmnist':
		morph = transforms.Compose([transforms.ToTensor()])
		test_data = datasets.FashionMNIST(root='../data', transform=morph, train=False, download=True)

	else:
		print("Unknown dataset. Help!")

	if args.use_seed:
		np.random.seed(args.seed)

	# Convert to batches for faster testing.
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
											shuffle=args.shuffle, num_workers=16)

	"""

	Add new networks here.

	"""
	if args.net=="resnet-cifar":
		print("Using CIFAR-10 pretrained ResNet-50")
		net = resnet50(pretrained=True) # CIFAR10 pretrained!
		net_2 = densenet121(pretrained=True)
		args.net_2 = 'densenet121'

	if args.net=="resnet18":
		print("Using CIFAR-10 pretrained ResNet-18")
		net = resnet18(pretrained=True) # CIFAR10 pretrained!

	if args.net=="resnet18-kernel-def":
		print("Using CIFAR-10 pretrained ResNet-18")
		net = resnet18(pretrained=True) # CIFAR10 pretrained!
		state_dict = net.state_dict()
		new_state_dict = OrderedDict()
		flag = False
		new_state_dict = OrderedDict()
		for k, data in state_dict.items():
			if args.rho > 0:
				if k == 'conv1.weight':
	            # if k.find('weight') != -1:
	            #     if k.find('conv1.') != -1:
					flag = True
					data = generateSmoothKernel(data.numpy(), args.rho)
					data = torch.from_numpy(data)
			new_state_dict[k] = data
		net.load_state_dict(new_state_dict)
		if not flag:
			print('Check conv1 layer name...')
			raise('lss')

	if args.net=="tiny-imagenet":
		print("Using Tiny-ImageNet pretrained ResNet-50")
		net = models.resnet50()
		net.fc = nn.Linear(2048, 200)
		net.load_state_dict(torch.load('../../drive/My Drive/tiny_imagenet/best_model_2.pth')['model_state_dict'])

	if args.net=="tiny-imagenet-kernel-def":
		print("Using Tiny-ImageNet pretrained ResNet-50, kernel-def")
		net = models.resnet50()
		net.fc = nn.Linear(2048, 200)
		net.load_state_dict(torch.load('../../drive/My Drive/tiny_imagenet/best_model_2.pth')['model_state_dict'])
		state_dict = net.state_dict()
		new_state_dict = OrderedDict()

		flag = False
		for k, data in state_dict.items():
			print(k)
			if args.rho > 0:
				if k == 'conv1.weight':
					flag = True
					print('Modifying', k)
					data = generateSmoothKernel(data.cpu().numpy(), args.rho)
					data = torch.from_numpy(data)
			new_state_dict[k] = data
		net.load_state_dict(new_state_dict)
		if not flag:
			print('Check conv1 layer name...')
			raise('lss')

		net_2 = models.resnet50()

	elif args.net=="resnet-cifar-kernel-def":
		print("Using CIFAR-10 pretrained ResNet-50")
		net = resnet50(pretrained=True) # CIFAR10 pretrained!
		state_dict = net.state_dict()
		new_state_dict = OrderedDict()
		for k, data in state_dict.items():
			if args.rho > 0:
				if k == 'conv1.weight':
					print('Modifying', k)
					data = generateSmoothKernel(data.cpu().numpy(), args.rho)
					data = torch.from_numpy(data)
			new_state_dict[k] = data
		net.load_state_dict(new_state_dict)

	elif args.net=="densenet121":
		print("Using CIFAR-10 pretrained DenseNet-121")
		net = densenet121(pretrained=True) # CIFAR10 pretrained!
		net_2 = resnet50(pretrained=True)
		args.net_2 = 'resnet-cifar'

	elif args.net=="densenet-kernel-def":
		print("Using CIFAR-10 pretrained ResNet-50")
		net_2 = resnet50(pretrained=True) # CIFAR10 pretrained!
		state_dict = net_2.state_dict()
		new_state_dict = OrderedDict()
		for k, data in state_dict.items():
			if args.rho > 0:
				if k == 'conv1.weight':
					print('Modifying', k)
					data = generateSmoothKernel(data.cpu().numpy(), args.rho)
					data = torch.from_numpy(data)
			new_state_dict[k] = data
		net_2.load_state_dict(new_state_dict)

		net = densenet121(pretrained=True)
		state_dict = net.state_dict()
		new_state_dict = OrderedDict()
		flag = False
		for k, data in state_dict.items():
			print(k)
			if args.rho > 0:
				if k == 'features.conv0.weight':
					flag = True
					print('Modifying', k)
					data = generateSmoothKernel(data.cpu().numpy(), args.rho)
					data = torch.from_numpy(data)
			new_state_dict[k] = data
		net.load_state_dict(new_state_dict)

	elif args.net=="resnet50-imagenet":
		print('Using ImageNet pretrained ResNet-50')
		net = torchvision.models.resnet50(pretrained=True)
		net = nn.Sequential(net, nn.Linear(in_features=1000, out_features=200, bias=True))
		net.load_state_dict(torch.load('../../drive/My Drive/tiny_imagenet/best_model.pth')['model_state_dict'])

	elif args.net=="resnet-madry":
		print("Using ResNet-50 with Madry training")
		from robustness.model_utils import make_and_restore_model
		from robustness.datasets import CIFAR
		ds = CIFAR('../data')
		net, _ = make_and_restore_model(parallel=False, arch='resnet50', dataset=ds, resume_path='./cifar_linf_8.pt')
		net = net.model

	elif args.net=="resnet-madry-kernel-def":
		print("Using ResNet-50 with Madry training")
		from robustness.model_utils import make_and_restore_model
		from robustness.datasets import CIFAR
		ds = CIFAR('../data')
		net, _ = make_and_restore_model(parallel=False, arch='resnet50', dataset=ds, resume_path='./cifar_linf_8.pt')
		net = net.model
		state_dict = net.state_dict()
		new_state_dict = OrderedDict()
		for k, data in state_dict.items():
			if args.rho > 0:
				if k == 'conv1.weight':
					print('Modifying', k)
					data = generateSmoothKernel(data.cpu().numpy(), args.rho)
					data = torch.from_numpy(data)
			new_state_dict[k] = data
		net.load_state_dict(new_state_dict)

	elif args.net=="imagenet-madry":
		print("Using ResNet-50 with Madry training")
		from robustness.model_utils import make_and_restore_model
		from robustness.datasets import ImageNet
		ds = ImageNet('../data')
		net, _ = make_and_restore_model(parallel=False, arch='resnet50', dataset=ds, resume_path='./imagenet_linf_8.pt')
		net = net.model

	elif args.net=="resnet-trades":
		print("Using WideResNet with TRADES training")
		from TRADES.models.wideresnet import WideResNet
		net = WideResNet()
		net.load_state_dict(torch.load('../../drive/My Drive/cifar/model_cifar_wrn.pt'))
		net.eval()
		for params in net.parameters():
			params.requires_grad = False
		net.cuda()

	elif args.net=="mnist-trades":
		print("Using Custom model with TRADES training")
		from TRADES.models.small_cnn import SmallCNN
		net = SmallCNN()
		net.load_state_dict(torch.load('../../drive/My Drive/mnist/model_mnist_smallcnn.pt'))
		net.eval()
		for params in net.parameters():
			params.requires_grad = False
		net.cuda()

	elif args.net=="mnist-feat-scat":
		print("Using WideResNet with Feature Scatter training")
		from FeatureScatter.models import *
		net = WideResNet(depth=28, num_classes=10, widen_factor=10)
		checkpoint = torch.load('../../drive/My Drive/mnist/latest')['net']
		re_check = dict()
		for i in checkpoint.keys():
			re_check[i[17:]] = checkpoint[i]
		net.load_state_dict(re_check)

	elif args.net=="resnet-feat-scat":
		print("Using WideResNet with Feature Scatter training")
		from FeatureScatter.models import *
		net = WideResNet(depth=28, num_classes=10, widen_factor=10)
		checkpoint = torch.load('../../drive/My Drive/cifar/feat_scat.pt')['net']
		re_check = dict()
		for i in checkpoint.keys():
			re_check[i[17:]] = checkpoint[i]
		net.load_state_dict(re_check)

	elif args.net=="inception":
		print("Using Inception Net")
		from facenet_pytorch import fixed_image_standardization, InceptionResnetV1
		net = InceptionResnetV1(pretrained='casia-webface', classify=True)
		net.logits = nn.Linear(512, 337)
		net.load_state_dict(torch.load('../../drive/My Drive/multi_pie_data/best_inception_model_4.pth')['model_state_dict'])

		from resnet50_scratch_dims_2048 import *
		net_2 = resnet50_scratch(weights_path='./resnet50_scratch_dims_2048.pth')
		net_2.classifier = nn.Conv2d(2048, 337, kernel_size=[1, 1], stride=(1, 1))
		net_2.load_state_dict(torch.load('../../drive/My Drive/multi_pie_data/best_model_3.pth')['model_state_dict'])
		morph_2 = transforms.Compose([transforms.Normalize(mean=[131.0912, 103.8827, 91.4953], std=[1, 1, 1])])

	elif args.net=="vggface2":
		print("Using Oxford VGG Net")
		from resnet50_scratch_dims_2048 import *
		net = resnet50_scratch(weights_path='./resnet50_scratch_dims_2048.pth')
		net.classifier = nn.Conv2d(2048, 337, kernel_size=[1, 1], stride=(1, 1))
		net.load_state_dict(torch.load('../../drive/My Drive/multi_pie_data/best_model_3.pth')['model_state_dict'])

	elif args.net=="vggface2-kernel-def":
		print("Using Oxford VGG Net, kernel-def")
		from resnet50_scratch_dims_2048 import *
		net = resnet50_scratch(weights_path='./resnet50_scratch_dims_2048.pth')
		net.classifier = nn.Conv2d(2048, 337, kernel_size=[1, 1], stride=(1, 1))
		net.load_state_dict(torch.load('../../drive/My Drive/multi_pie_data/best_model_3.pth')['model_state_dict'])
		state_dict = net.state_dict()
		new_state_dict = OrderedDict()
		flag = False
		for k, data in state_dict.items():
			print(k)
			if args.rho > 0:
				if k == 'conv1_7x7_s2.weight':
					flag = True
					print('Modifying', k)
					data = generateSmoothKernel(data.cpu().numpy(), args.rho)
					data = torch.from_numpy(data)
			new_state_dict[k] = data
		net.load_state_dict(new_state_dict)
		if not flag:
			print('Check conv1 layer name...')
			raise('lss')

	elif args.net=="fmnist":
		print("Using fmnist ")
		class Flatten(torch.nn.Module):
			def __init__(self):
				super(Flatten, self).__init__()

			def forward(self, x):
				return x.view(x.size(0), -1)

		net = nn.Sequential(
			nn.BatchNorm2d(1, affine=False),
			nn.Conv2d(1, 10, kernel_size=5),
			nn.MaxPool2d(2),
			nn.ReLU(),
			nn.Conv2d(10, 20, kernel_size=5),
			nn.MaxPool2d(2),
			nn.ReLU(),
			nn.Dropout2d(p=0.2),
			Flatten(),
			nn.Linear(320, 10))

		net.load_state_dict(torch.load('../../drive/My Drive/fmnist/3.0-fb-conv-classifier.pth'))

	elif args.net=="fmnist-kernel-def":
		print("Using fmnist ")
		class Flatten(torch.nn.Module):
			def __init__(self):
				super(Flatten, self).__init__()

			def forward(self, x):
				return x.view(x.size(0), -1)

		net = nn.Sequential(
			nn.BatchNorm2d(1, affine=False),
			nn.Conv2d(1, 10, kernel_size=5),
			nn.MaxPool2d(2),
			nn.ReLU(),
			nn.Conv2d(10, 20, kernel_size=5),
			nn.MaxPool2d(2),
			nn.ReLU(),
			nn.Dropout2d(p=0.2),
			Flatten(),
			nn.Linear(320, 10))

		net.load_state_dict(torch.load('../../drive/My Drive/fmnist/3.0-fb-conv-classifier.pth'))

		state_dict = net.state_dict()
		new_state_dict = OrderedDict()
		for k, data in state_dict.items():
			print(k)
			if args.rho > 0:
				if k == '1.weight':
					print('Modifying', k)
					data = generateSmoothKernel(data.cpu().numpy(), args.rho)
					data = torch.from_numpy(data)
			new_state_dict[k] = data
		net.load_state_dict(new_state_dict)

	elif args.net=="imagenet":
		net = torchvision.models.resnet50(pretrained=True)

	if args.defence_type=="comdefend":
		pass

	print("Freezing network layers and setting to eval mode")
	net.eval()
	for params in net.parameters():
		params.requires_grad = False

	# net_2.eval()
	# for params in net_2.parameters():
	# 	params.requires_grad = False


	print("Testing on:", args.dataset)

	if args.gaussian_smoothing:
		print('Using gaussian smoothing')	

	if args.cuda:
		net.cuda()
		# net_2.cuda()

	test_iter = iter(test_loader)
	print(len(test_loader.dataset))
	test_len = len(test_loader.dataset)
	total = len(test_loader.dataset)

	miss_correct = 0 		# Misclassifies as target.
	miss_incorrect = 0 		# Misclassifies, but not as target.
	correct = 0 			# Classifies correctly.

	uiqi_score = 0.0
	ssim_score = 0.0

	"""
	
	Main attack loop.
	Key arguments:
		1. sgs: Used when testing model accuracy without any attack.
		2. pert_img: The perturbed image batch obtained from whichever attack was used. 

	"""

	savefile = dict()
	# IMAGES = np.ndarray()
	# LABELS = np.ndarray()
	print(args.shuffle)
	if not args.shuffle:
		print("not shuffling!")

	for i, data in enumerate(test_loader):

		# if i < 5000:
		# 	continue

		# i -= 5000

		img, label = data
		img, label = img.cuda(), label.cuda()

		sgs = img.clone().detach()
		sgs.requires_grad = False

		output_normal = net(sgs)
		_, predicted_normal = torch.max(output_normal, 1)	

		if args.dataset=='imagenet':
			for q in range(len(predicted_normal)):
				# print(q)
				predicted_normal[q] = test_data.class_to_idx[str(predicted_normal[q].data.cpu().numpy())]

			predicted_normal = predicted_normal.cuda()

		if args.attack_type=='max_error' or args.attack_type=='none':
			target = label.clone().detach()
		
		# target = predicted_normal.clone().detach()

		if args.attack_type!='none':	
			pert_img = perturb_img_bands_only(img.clone().detach(), label, target.clone().detach(), net, args.attack_type, 
								args.dataset, args.net, random_restarts=args.random_restarts,
								bands=args.bands, update_type=args.update_type, odi=args.odi,
								loss_type=args.loss_type, class2idx=test_data.class_to_idx)
		else:
			pert_img = sgs

		output = net(pert_img.clone().detach())
		# output = net_2(pert_img.clone().detach())
			

		_, predicted = torch.max(output, 1)
		
		if args.dataset=='imagenet':
			for q in range(len(predicted)):
				predicted[q] = test_data.class_to_idx[str(predicted[q].data.cpu().numpy())]

			predicted = predicted.cuda()

		print('predicted:', predicted)
		print('predicted_normal:', predicted_normal)
		print('label:', label)
		"""

		Calculate classification rates. 

		"""
		miss_correct+=(predicted==target).sum().item()
		for k, j in zip(torch.eq(predicted, predicted_normal).tolist(), torch.eq(predicted, predicted_normal).tolist()):
			miss_incorrect+=(not(k or j))

		correct+=(predicted==label).sum().item()
		
		"""

		UIQI scoring. Using the libary sewar here.

		"""
		# with torch.no_grad():
		# 	uiqi_score += get_uiqi(postprocess(img.clone(), args.dataset, args.net), 
		# 						   postprocess(pert_img.clone(), args.dataset, args.net))

		"""

		Save all images.		

		"""
		with torch.no_grad():
			try:
				LABELS = np.concatenate((LABELS, label.data.cpu().numpy()), axis=None)
				IMAGES = np.concatenate((IMAGES, postprocess(pert_img.clone(), args.dataset, args.net).data.cpu().numpy()), axis=0)
				print(LABELS.shape)
				print(IMAGES.shape)
			except:
				LABELS = label.data.cpu().numpy()
				IMAGES = postprocess(pert_img.clone(), args.dataset, args.net).data.cpu().numpy()
				print(LABELS.shape)
				print(IMAGES.shape)

		"""

		Sample images.

		"""

		if (i+1)*args.batch_size%(args.batch_size)==0:
			if args.gaussian_smoothing:
				side_plot(img.clone(), gauss_pert_img.clone(), label, predicted, i, args.dataset, args.net)
				gauss_side_plot(img, pert_img, gauss_pert_img, args.dataset, i, args.net)
			else:
				if (i+1)*args.batch_size%(100*args.batch_size)==0:
					print(F.softmax(output, dim=1))
					print(F.softmax(output_normal, dim=1))
					side_plot(img.clone(), pert_img.clone(), label, predicted, i, args.dataset, args.net)
					# break
			print("\nMisclassified as target:\t\t", miss_correct/((i+1)*args.batch_size))
			print("Misclassified, but not as target:\t", miss_incorrect/((i+1)*args.batch_size))
			print("Network accuracy on perturbed test data:", correct/((i+1)*args.batch_size))
			print("Average UIQI score:\t\t\t", uiqi_score/((i+1)*args.batch_size))
			# print("Average SSIm score:\t\t\t", ssim_score/((i+1)*args.batch_size))
			print("Processed:", (i+1)*args.batch_size, i+1, args.batch_size)

	print(LABELS)
	print(IMAGES)

	savefile["images"] = IMAGES
	savefile["labels"] = LABELS

	savemat('cifar-10.mat', savefile)

	print("Network accuracy on perturbed test data:", correct/total)