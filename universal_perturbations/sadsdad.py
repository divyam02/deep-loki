from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from utils.models import *

morph = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
									std=[0.2023, 0.1994, 0.2010])])

test_data = datasets.CIFAR10(root='../data/', transform=morph, train=False, download=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=250, shuffle=False)

net = resnet50(pretrained=True)
net = net
net.eval()

test_iter = iter(test_loader)
print(len(test_loader.dataset))
test_len = len(test_loader.dataset)//250
correct = 0
for i in range(test_len):
	with torch.no_grad():
		img, label = next(test_iter)
		img = img
		output = net(img) # Use perturbed image!
		_, predicted = torch.max(output, 1)
		print(predicted, label)
		correct += (predicted==label).sum().item()
			

print("Network accuracy on perturbed test data:", 100 * correct/10000)
