import torch
import torch.nn as nn
import torch.transforms as transforms
import numpy as np

BINARY_SEARCH_STEPS = 9
MAX_ITERATIONS = 10000
EARLY_ABORT = True
LEARNING_RATE = 1e-3
TARGETED = True
CONFIDENCE = 0
INITIAL_CONST = 1e-3
BOX_MIN = -0.5
BOX_MAX = 0.5

class L2_attack:
	def __init__(self, train_loader, val_loader, target_labels, 
				model, batch_size=1, targeted=TARGETED, lr=LEARNING_RATE,
				bin_search_steps=BINARY_SEARCH_STEPS, max_iter=MAX_ITERATIONS,
				early_abort=EARLY_ABORT, init_const=INITIAL_CONST,
				box_min=BOX_MIN, box_max=BOX_MAX):

		self.model = model
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.target_labels = target_labels
		self.batch_size = batch_size
		self.targeted = targeted
		self.lr = lr
		self.bin_search_steps = bin_search_steps
		self.max_iter = max_iter
		self.early_abort = early_abort
		self.init_const = init_const
		self.box_max = box_max
		self.box_min = box_min
		self.init_const = init_const

		image_size, num_channels, num_labels = 32, 3, 10
		self.repeat = BINARY_SEARCH_STEPS >= 10
		shape=(batch_size, num_channels, image_size, image_size)

		modifier = torch.from_numpy(np.zeros(shape)).cuda()
		print(modifier, modifier.shape)

		input("continue?")

		self.timg = torch.from_numpy(np.zeros(shape), dtype=torch.float32).cuda()
		self.tlab = torch.from_numpy(np.zeros(batch_size, num_labels)).cuda()
		self.const = torch.from_numpy(np.zeros(shape), dtype=torch.float32)

		self.box_mul = (self.box_max - self.box_min) / 2
		self.box_plus = (self.box_min + self.box_max) / 2
		self.newimg = nn.tanh(modifier+self.timg)  * self.box_mul + self.box_plus

		self.output = self.model.forward(self.newimg)

		self.l2dist = 0


	def l2_attack(train_loader, target_labels, model):
		
		pass