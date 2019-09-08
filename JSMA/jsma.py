import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def get_forward_derivative(model, img, target):
	"""
	Return forward-derivative of target
	class wrt. all img inputs

	Return gradients vector with img shape.
	"""
	img.requires_grad_()
	# img.grad.data.zero_()
	# print(img)
	# input()
	output = model.forward(img)
	model.zero_grad()
	output[0, target].backward(retain_graph=True)

	req_grads = img.grad.clone()
	temp = 0
	for i in range(10):
		if i!=target:
			img.grad.data.zero_()
			# print(img.grad.data)
			model.zero_grad()
			output[0, i].backward(retain_graph=True)
			temp += img.grad.data
			temp = temp.data.clone()
			# print("temp", temp)
			# input("continue?")

	return req_grads.data, temp.data

def get_saliency_map_max_i(model, img, target):
	target_grads, sum_grads = get_forward_derivative(model, img, target)
	saliency_map = torch.zeros(target_grads.size())

	"""
	Define measure values. Assign  
	"""

	max_i = -np.inf
	max_index = None
	for i in range(len(target_grads[0])):
		for j in range(len(target_grads[0, i])):
			for k in range(len(target_grads[0, i, j])):
				# print("sum_grads", sum_grads[0, i, j, k])
				# print("target_grads", target_grads[0, i, j, k])
				if target_grads[0, i, j, k].data < 0 or sum_grads[0, i, j, k].data > 0:
					saliency_map[0, i, j, k] = 0
				else:
					saliency_map[0, i, j, k] = target_grads[0, i, j, k].data * abs(sum_grads[0, i, j, k].data)


				if saliency_map[0, i, j, k].data > max_i:
					max_i = saliency_map[0, i, j, k].data
					max_index = (0, i, j, k)

				# print("curr value:", saliency_map[0, i, j, k])
				# print("max_i:", max_i)
				# print("max_index:", max_index)
				# print(saliency_map)
				# input()

	print("max index:", max_index)
	return max_index


def perturb_img(img, target, model, max_iters=1000,
				theta=0.1):
	
	iters = 0
	predicted = None
	pert_img = img.clone().cuda()
	while predicted!=target and iters<max_iters:
		# Send img copy!
		a, b, c, d = get_saliency_map_max_i(model, pert_img.clone(), target)
		print(pert_img[a, b, c, d])
		pert_img[a, b, c, d] += theta
		print(pert_img[a, b, c, d])
		
		with torch.no_grad():
			pert_img = pert_img.cuda()
			output = model(pert_img)
			_, predicted = torch.max(output.data, 1)

			if predicted==target:
				print("success!")
		iters+=1
		print("iteration", iters)

	return pert_img
