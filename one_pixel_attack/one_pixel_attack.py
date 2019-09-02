from torchvision import models
import torch
import cv2
import numpy as np
from scipy.optimize import differential_evolution
import torch.nn as nn
from torch.autograd import Variable
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.sampler as sampler
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from utils.models import *

cifar10_class_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, 
					default='airplane.png', 
					help='path to image')
parser.add_argument('--d', type=int, default=1,
					 help='number of pixels to change')
parser.add_argument('--iters', type=int, default=600, 
					help='number of iterations')
parser.add_argument('--popsize', type=int, default=10, 
					help='population size')
parser.add_argument('--model_path', type=str, default='cifar10_basiccnn.pth.tar', 
					help='path to trained model')
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
image_path = args.img
d = args.d
iters = args.iters
popsize = args.popsize

def nothing(x):
    pass

"""
orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (32, 32))
img = orig.copy()
shape = orig.shape
"""
morph = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
									std=[0.2023, 0.1994, 0.2010])])
all_data = datasets.CIFAR10(root = '../data/', transform=morph, train=True, download=True)
test_data = datasets.CIFAR10(root='../data/', transform=morph, train=False, download=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
											shuffle=True)
test_iter = iter(test_loader)

if args.net=="resnet50":
	print("Using ResNet-50")
	model = resnet50(pretrained=True) # CIFAR10 pretrained!

elif args.net=="densenet121":
	print("Using DenseNet-121")
	model = densenet121(pretrained=True) # CIFAR10 pretrained!

model.eval()

if args.cuda:
	model.cuda()


def revert(pert_img):
	pert_img[0][0] *= 0.2023
	pert_img[0][1] *= 0.1994
	pert_img[0][2] *= 0.2010
	pert_img[0][0] += 0.4914
	pert_img[0][1] += 0.4822
	pert_img[0][2] += 0.4465

	pert_img = pert_img.detach().cpu().squeeze().numpy()

	return pert_img.transpose(1, 2, 0)

def preprocess(pert_img):
	pert_img = pert_img.transpose(2, 0, 1)

	pert_img[0][0] -= 0.4914
	pert_img[0][1] -= 0.4822
	pert_img[0][2] -= 0.4465
	pert_img[0][0] /= 0.2023
	pert_img[0][1] /= 0.1994
	pert_img[0][2] /= 0.2010

	return pert_img


counter = 0
img, label = next(test_iter)
shape = revert(img).shape

for i in range(len(test_loader.dataset)-1):
	img, label = next(test_iter)
	img, label = img.cuda(), label.cuda()

	def softmax(x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()


	"""
	model = BasicCNN()
	saved = torch.load(model_path, map_location='cpu')
	model.load_state_dict(saved['state_dict'])
	model.eval()
	"""
	inp = Variable(img).float().cuda()
	prob_orig = softmax(model(inp).detach().cpu().data.numpy()[0])
	pred_orig = np.argmax(prob_orig)
	print('Prediction before attack: %s' %(cifar10_class_names[pred_orig]))
	print('Probability: %f' %(prob_orig[pred_orig]))

	def perturb(x):
		adv_img = revert(img).copy()

		# calculate pixel locations and values
		pixs = np.array(np.split(x, len(x)/5)).astype(int)
		loc = (pixs[:, 0], pixs[:,1])
		val = pixs[:, 2:]
		adv_img[loc] = val

		return adv_img

	def optimize(x):
		adv_img = perturb(x)

		#print("adv_img", type(preprocess(adv_img)))
		inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0)).cuda()
		out = model(inp)
		prob = softmax(out.detach().cpu().data.numpy()[0])

		return prob[pred_orig]
		pred_adv = 0
		prob_adv = 0

	def callback(x, convergence):
	    global pred_adv, prob_adv
	    adv_img = perturb(x)

	    inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0)).cuda()
	    out = model(inp)
	    prob = softmax(out.detach().cpu().data.numpy()[0])

	    pred_adv = np.argmax(prob)
	    prob_adv = prob[pred_adv]
	    if pred_adv != pred_orig and prob_adv >= 0.9:
	        print('Attack successful..')
	        print('Prob [%s]: %f' %(cifar10_class_names[pred_adv], prob_adv))
	        return True


	def scale(x, scale=5):
	    return cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)



#	while True:
	bounds = [(0, shape[0]-1), (0, shape[1]), (0, 1), (0, 1), (0, 1)] * d
	result = differential_evolution(optimize, bounds, maxiter=iters, popsize=popsize, tol=1e-5, callback=callback)

	adv_img = perturb(result.x)
	inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0)).cuda()
	out = model(inp)
	prob = softmax(out.detach().cpu().data.numpy()[0])
	print('Prob [%s]: %f --> Prob[%s]: %f' %(cifar10_class_names[pred_orig], prob_orig[pred_orig], cifar10_class_names[pred_adv], prob_adv), "\n")

	if counter%2500==0:
		print("saving image!")
		cv2.imwrite('adv_img.png'+str(counter), scale(adv_img[..., ::-1]))

	counter += 1
