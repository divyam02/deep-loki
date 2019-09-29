import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.sampler as sampler
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from runutils import *
import sys
from utils.models import *
from msb import *



def big_plot(og_img, pert_imgs, label, k_is, i):
	"""
	Quick visual debug!
	"""
	_, ax = plt.subplots(nrows=1, ncols=11)
	# plt.figure(figsize=(10, 110))

	classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	#og_img = inv_transform(og_img)
	og_img[0][0] *= 0.2023
	og_img[0][1] *= 0.1994
	og_img[0][2] *= 0.2010
	og_img[0][0] += 0.4914
	og_img[0][1] += 0.4822
	og_img[0][2] += 0.4465

	#pert_img = inv_transform(pert_img)

	ax[0].imshow(og_img.data.cpu().squeeze().permute(1, 2, 0))
	ax[0].set_title(classes[label])

	for i in range(len(pert_imgs)):
		pert_img = pert_imgs[i]
		k_i = k_is[i]
		pert_img[0][0] *= 0.2023
		pert_img[0][1] *= 0.1994
		pert_img[0][2] *= 0.2010
		pert_img[0][0] += 0.4914
		pert_img[0][1] += 0.4822
		pert_img[0][2] += 0.4465		

		ax[i+1].imshow(pert_img.data.cpu().squeeze().permute(1, 2, 0))
		ax[i+1].set_title(classes[k_i])
	plt.savefig('works_'+str(i)+'.png')

def side_plot(og_img, pert_img, label, k_i, i):
	"""
	Quick visual debug!
	"""
	_, ax = plt.subplots(2)

	classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	np.save('img_'+str(i), og_img.data.cpu().numpy())
	np.save('adv_'+str(i), pert_img.data.cpu().numpy())

	#og_img = inv_transform(og_img)
	og_img[0][0] *= 0.2023
	og_img[0][1] *= 0.1994
	og_img[0][2] *= 0.2010
	og_img[0][0] += 0.4914
	og_img[0][1] += 0.4822
	og_img[0][2] += 0.4465

	pert_img[0][0] *= 0.2023
	pert_img[0][1] *= 0.1994
	pert_img[0][2] *= 0.2010
	pert_img[0][0] += 0.4914
	pert_img[0][1] += 0.4822
	pert_img[0][2] += 0.4465

	#pert_img = inv_transform(pert_img)
	# cv2.imwrite('img_'+str(i)+'.jpg', cv2.cvtColor(og_img[0].data.cpu().numpy(), cv2.COLOR_RGB2BGR))
	# cv2.imwrite('adv_'+str(i)+'.jpg', cv2.cvtColor(pert_img[0].data.cpu().numpy(), cv2.COLOR_RGB2BGR))

	ax[0].imshow(og_img.data.cpu().squeeze().permute(1, 2, 0))
	ax[1].imshow(pert_img.data.cpu().squeeze().permute(1, 2, 0))
	ax[0].set_title(classes[label])
	ax[1].set_title(classes[k_i])
	plt.savefig('works_'+str(i)+'.png')

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

	# Take MSB fraction from user... 
	parser.add_argument('--fraction', dest='fraction',
						type=float, default=0.2)

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
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
											shuffle=True)

	if args.net=="resnet50":
		print("Using ResNet-50")
		net = resnet50(pretrained=True) # CIFAR10 pretrained!

	elif args.net=="densenet121":
		print("Using DenseNet-121")
		net = densenet121(pretrained=True) # CIFAR10 pretrained!

	net.eval()

	if args.cuda:
		net.cuda()

	mean = [0.4914, 0.4822, 0.4465]
	std = [0.2023, 0.1994, 0.2010]
	inputs_box = (min((0 - m) / s for m, s in zip(mean, std)), 
				  max((1 - m) / s for m, s in zip(mean, std)))

	adversary = L2Adversary(targeted=True, confidence=0.0,
							search_steps=10, box=inputs_box,
							optimizer_lr=5e-4)

	test_iter = iter(test_loader)
	print(len(test_loader.dataset))
	test_len = len(test_loader.dataset)//100
	total = len(test_loader.dataset)//100
	correct = 0

	for i in range(test_len):
		img, label = next(test_iter)
		# if choice_imgs[label]==1:
		# 	continue

		# choice_imgs[label] = 1
		# print("label choice:", label)
		# save_image(img[0], 'img_label_'+str(label)+'.jpg')
		img, label = img.cuda(), label.cuda()
		# for j in range(10):
		attack_targets = torch.ones(img.size(0)).cuda() * 9
		if label==9:
			attack_targets = torch.ones(img.size(0)).cuda() * 0	
		adversarial_examples = add_msb_noise(net, img, args.fraction, to_numpy=False)
		output_real = net(img)
		_, pred_real = torch.max(output_real, 1)
		output = net(adversarial_examples.cuda()) # Use perturbed image!
		_, predicted = torch.max(output, 1)

			# preds.append(predicted)
		print("Label:", label, "Real Predicted", pred_real, "Pert Predicted:", predicted, "Iter:", i)
		correct += (predicted==pred_real).sum().item()

		# save_image(adversarial_examples[0], 'adv_label_'+str(label)+'.jpg')

		# print(adversarial_examples.shape)
		# print(type(adversarial_examples))
		# side_plot(img, adversarial_examples, label, int(attack_targets.item()))
		# input("continu?")
		if i%25==0:
			side_plot(img, adversarial_examples, pred_real, predicted, label.data)
			print("Network accuracy on perturbed test data:", correct/(i+1))

		# if sum(choice_imgs)==10:
		# 	break

	# big_plot(img, adversarial_examples, pred_real, preds, 3000)

	print("Network accuracy on perturbed test data:", correct/total)