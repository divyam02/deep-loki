import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_wavelets import DWTForward, DWTInverse
import pywt
import pywt.data
from torchvision import transforms as transforms
import torch.optim as optim
import copy
from torch.autograd import Variable
"""

Attack methods.

"""
def perturb_img_fgsm(img, label, target, model, attack_type, dataset=None, net_type=None, alpha=0,
				max_iters=20, epsilon=8.0/255.0, random_restarts=False, update_type='gradient',
				step_size=2.0/255.0, bands='ll', lr=5e-2, odi=False, loss_type='xent', class2idx=None):
	print('Using FGSM...')
	print('Loss fn:', loss_type)
	print('Update:', update_type)

	# idx2class = dict()
	# for key in class2idx:
	# 	idx2class[str(class2idx[key])] = int(key)

	if random_restarts: num_restarts=20
	else: num_restarts = 1

	odi_step_size = epsilon
	best_adv = None
	curr_pred = None

	if dataset=='imagenet':
		for i in range(len(target)):
			# print('before', target[i], idx2class[str(target[i].data.cpu().numpy())])
			target[i] = idx2class[str(target[i].data.cpu().numpy())]
			# pritn('after', target[i])

		target = target.cuda()

	# avg_Linf_error = 0

	adv = Variable(postprocess(img.clone().detach(), dataset, net_type).data, requires_grad=True)

	if loss_type=='xent':
		loss_fn = torch.nn.CrossEntropyLoss()
	elif loss_type=='margin':
		loss_fn = margin_loss

	# for step in range(max_iters):

	optimizer = optim.SGD([adv], lr=1e-3)
	optimizer.zero_grad()
	pred = model(preprocess(adv, dataset, net_type))

	loss = loss_fn(pred, target)
	loss.backward(retain_graph=True)
	print(target)
	# input('continue')
	eta = epsilon * adv.grad.data.sign()

	adv = Variable(adv.data + eta, requires_grad=True)
	eta = torch.clamp(adv.data - postprocess(img.clone().detach(), dataset, net_type), -epsilon, epsilon)
	adv = Variable(postprocess(img.clone().detach(), dataset, net_type) + eta, requires_grad=True)
	if dataset=='multi-pie':
		adv = Variable(torch.clamp(adv, 0, 255.0), requires_grad=True)
	else:
		adv = Variable(torch.clamp(adv, 0, 1.0), requires_grad=True)

	with torch.no_grad():
		curr_pred = torch.max(model(preprocess(adv, dataset, net_type)), 1)[1]
		if best_adv==None:
			best_adv = adv.clone().detach()
		else:
			for i in range(len(curr_pred)):
				if curr_pred[i].data != target[i].data:
					best_adv[i] = adv[i]

	return preprocess(best_adv, dataset, net_type)



def perturb_img_pgd(img, label, target, model, attack_type, dataset=None, net_type=None, alpha=0,
				max_iters=20, epsilon=8.0/255.0, random_restarts=False, update_type='gradient',
				step_size=2.0/255.0, bands='ll', lr=5e-2, odi=False, loss_type='xent', class2idx=None):

	print('using PGD')

	print('Loss fn:', loss_type)
	print('Update:', update_type)


	# idx2class = dict()
	# for key in class2idx:
	# 	idx2class[str(class2idx[key])] = int(key)

	if dataset=='imagenet':
		for i in range(len(target)):
			# print('before', target[i], idx2class[str(target[i].data.cpu().numpy())])
			target[i] = idx2class[str(target[i].data.cpu().numpy())]
			# pritn('after', target[i])

		target = target.cuda()

	print(target)

	if random_restarts: num_restarts=20
	else: num_restarts = 1

	best_adv = None
	curr_pred = None

	# avg_Linf_error = 0

	for restart in range(num_restarts):
		random_init = random_restarts
		print('Restart {}'.format(restart+1))

		adv = Variable(postprocess(img.clone().detach(), dataset, net_type).data, requires_grad=True)
		if loss_type=='xent':
			loss_fn = torch.nn.CrossEntropyLoss()
		elif loss_type=='margin':
			loss_fn = margin_loss

		if random_init:
			random_noise = torch.FloatTensor(*adv.shape).uniform_(-epsilon, epsilon).cuda()
			adv = Variable(adv.data + random_noise, requires_grad=True)

		for step in range(max_iters):

			optimizer = optim.SGD([adv], lr=1e-3)
			optimizer.zero_grad()
			pred = model(preprocess(adv, dataset, net_type))
			
			loss = loss_fn(pred, target)
			loss.backward(retain_graph=True)

			eta = step_size * adv.grad.data.sign()

			adv = Variable(adv.data + eta, requires_grad=True)
			eta = torch.clamp(adv.data - postprocess(img.clone().detach(), dataset, net_type), -epsilon, epsilon)
			adv = Variable(postprocess(img.clone().detach(), dataset, net_type) + eta, requires_grad=True)
			if dataset=='multi-pie':
				adv = Variable(torch.clamp(adv, 0, 255.0), requires_grad=True)
			else:
				adv = Variable(torch.clamp(adv, 0, 1.0), requires_grad=True)

		with torch.no_grad():
			curr_pred = torch.max(model(preprocess(adv, dataset, net_type)), 1)[1]
			if best_adv==None:
				best_adv = adv.clone().detach()
			else:
				for i in range(len(curr_pred)):
					if curr_pred[i].data != target[i].data:
						best_adv[i] = adv[i]

	# print('Avg. L2 error:', torch.norm(best_adv - img, 2)/len(img))

	return preprocess(best_adv, dataset, net_type)


def perturb_img_bands_only(img, label, target, model, attack_type, dataset=None, net_type=None, alpha=0,
				max_iters=20, epsilon=8.0/255.0, random_restarts=False, update_type='gradient',
				step_size=2.0/255.0, bands='ll', lr=5e-2, odi=False, loss_type='xent', class2idx=None):
	
	if bands:
		print('Using bands only:', bands)

	print('Loss fn:', loss_type)
	print('Update:', update_type)


	# idx2class = dict()
	# for key in class2idx:
	# 	idx2class[str(class2idx[key])] = int(key)

	if dataset=='imagenet':
		for i in range(len(target)):
			# print('before', target[i], idx2class[str(target[i].data.cpu().numpy())])
			target[i] = idx2class[str(target[i].data.cpu().numpy())]
			# pritn('after', target[i])

		target = target.cuda()

	print(target)

	if random_restarts: num_restarts=20
	else: num_restarts = 1
	if odi: odi_steps = 2
	else: odi_steps = 0 

	if dataset=='multi-pie' and net_type=='vggface2':
		epsilon=8.0
		lr = 2.0

	if dataset=='multi-pie' and net_type=='vggface2-kernel-def':
		print('adjusted epsilon')
		epsilon=8.0
		lr = 2.0


	odi_step_size = epsilon
	best_adv = None
	curr_pred = None
	switch_band = bands

	ifm = DWTInverse(mode='zero', wave='haar').cuda()
	xfm = DWTForward(J=1, mode='zero', wave='haar').cuda()

	# avg_Linf_error = 0

	for restart in range(num_restarts):
		random_init = random_restarts
		print('Restart {}'.format(restart+1))

		adv = Variable(postprocess(img.clone().detach(), dataset, net_type).data, requires_grad=True)
		if loss_type=='xent':
			loss_fn = torch.nn.CrossEntropyLoss()
		elif loss_type=='margin':
			loss_fn = margin_loss
		if random_init:
			random_noise = torch.FloatTensor(*adv.shape).uniform_(-epsilon, epsilon).cuda()
			adv = Variable(adv.data + random_noise, requires_grad=True)

		for step in range(odi_steps + max_iters):
			LL, Y = xfm(adv)
			LH, HL, HH = torch.unbind(Y[0], dim=2)
			LL = Variable(LL.data, requires_grad=True)
			LH = Variable(LH.data, requires_grad=True)
			HL = Variable(HL.data, requires_grad=True)
			HH = Variable(HH.data, requires_grad=True)
			if bands=='ll':
				band_optim = optim.SGD([LL], lr=lr)
			elif bands=='lh':
				band_optim = optim.SGD([LH], lr=lr)
			elif bands=='hl':
				band_optim = optim.SGD([HL], lr=lr)
			elif bands=='hh':
				band_optim = optim.SGD([HH], lr=lr)
			elif bands=='high':
				band_optim = optim.SGD([LH, HL, HH], lr=lr)
			elif bands=='all':
				band_optim = optim.SGD([LL, LH, HL, HH], lr=lr)
			band_optim.zero_grad()
			adv = Variable(ifm((LL, [torch.stack((LH, HL, HH), 2)])).data, requires_grad=True)

			band_loss = loss_fn(model(preprocess(ifm((LL, [torch.stack((LH, HL, HH), 2)])), dataset, net_type)), target)
			band_loss.backward()
			temp = []

			if bands=='ll':
				for band in [LL]:
					band_eta = lr * band.grad.data.sign()
					band = Variable(band.data + band_eta, requires_grad=True)
					temp.append(band)
				LL, LH, HL, HH = temp[0], LH, HL ,HH
			elif bands=='lh':
				for band in [LH]:
					band_eta = lr * band.grad.data.sign()
					band = Variable(band.data + band_eta, requires_grad=True)
					temp.append(band)
				LL, LH, HL, HH = LL, temp[0], HL ,HH
			elif bands=='hl':
				for band in [HL]:
					band_eta = lr * band.grad.data.sign()
					band = Variable(band.data + band_eta, requires_grad=True)
					temp.append(band)
				LL, LH, HL, HH = LL, LH ,temp[0], HH
			elif bands=='hh':
				for band in [HH]:
					band_eta = lr * band.grad.data.sign()
					band = Variable(band.data + band_eta, requires_grad=True)	
					temp.append(band)
				LL, LH, HL, HH = LL, LH , HL, temp[0]
			elif bands=='high':
				for band in [LH, HL, HH]:
					band_eta = lr * band.grad.data.sign()
					band = Variable(band.data + band_eta, requires_grad=True)
					temp.append(band)
				LL, LH, HL, HH = LL, temp[0], temp[1], temp[2]
			elif bands=='all':
				for band in [LL, LH, HL, HH]:
					band_eta = lr * band.grad.data.sign()
					band = Variable(band.data + band_eta, requires_grad=True)
					temp.append(band)
				LL, LH, HL, HH = temp[0], temp[1], temp[2], temp[3]

			adv = ifm((LL, [torch.stack((LH, HL, HH), 2)]))

			eta = torch.clamp(adv.data - postprocess(img.clone().detach(), dataset, net_type), -epsilon, epsilon)
			adv = Variable(postprocess(img.clone().detach(), dataset, net_type) + eta, requires_grad=True)
			if dataset=='multi-pie' and net_type=='vggface2':
				adv = Variable(torch.clamp(adv, 0, 255.0), requires_grad=True)
			elif dataset=='multi-pie' and net_type=='vggface2-kernel-def':
				adv = Variable(torch.clamp(adv, 0, 255.0), requires_grad=True)
			else:
				adv = Variable(torch.clamp(adv, 0, 1.0), requires_grad=True)

		with torch.no_grad():
			curr_pred = torch.max(model(preprocess(adv, dataset, net_type)), 1)[1]
			if best_adv==None:
				best_adv = adv.clone().detach()
			else:
				for i in range(len(curr_pred)):
					if curr_pred[i].data != target[i].data:
						best_adv[i] = adv[i]

	return preprocess(best_adv, dataset, net_type)

"""

Utility functions.

"""
def get_bands(bands, LL, LH, HL, HH):
	LL = Variable(LL.data, requires_grad=True)
	LH = Variable(LH.data, requires_grad=True)
	HL = Variable(HL.data, requires_grad=True)
	HH = Variable(HH.data, requires_grad=True)
	if bands=='ll':
		return (LL)
	elif bands=='lh':
		return (LH)
	elif bands =='HL':
		return (HL)
	elif bands=='hh':
		return (HH)
	elif bands=='high':
		return (LH, HL, HH)
	elif bands=='all':
		return (LL, LH, HL, HH)

def preprocess(img, dataset=None, net_type=None):
	"""

	Preprocess by normalization. 
	
	"""
	temp = img.clone()

	if net_type=="inception":
		temp = (temp - 127.5)/128.0
	
	elif dataset=="multi-pie":
		temp[:, 0] -= 131.0912
		temp[:, 1] -= 103.8827
		temp[:, 2] -= 91.4953
		temp[:, 0] /= 1.0
		temp[:, 1] /= 1.0
		temp[:, 2] /= 1.0

	elif dataset=="fmnist":
		pass

	elif dataset=="tiny-imagenet":
		temp[:, 0] -= 0.4802
		temp[:, 1] -= 0.4481
		temp[:, 2] -= 0.3975
		temp[:, 0] /= 0.2302
		temp[:, 1] /= 0.2265
		temp[:, 2] /= 0.2262
	elif dataset=="cifar10-trades":
		pass
	elif dataset=='mnist':
		pass
	elif dataset=="cifar10-feat-scat":
		temp[:, 0] -= 0.5
		temp[:, 1] -= 0.5
		temp[:, 2] -= 0.5
		temp[:, 0] /= 0.5
		temp[:, 1] /= 0.5
		temp[:, 2] /= 0.5
	elif dataset=="imagenet":
		temp[:, 0] -= 0.485
		temp[:, 1] -= 0.456
		temp[:, 2] -= 0.406
		temp[:, 0] /= 0.229
		temp[:, 1] /= 0.224
		temp[:, 2] /= 0.225
	elif dataset=="cifar10":
		temp[:, 0] -= 0.4914
		temp[:, 1] -= 0.4822
		temp[:, 2] -= 0.4465
		temp[:, 0] /= 0.2023
		temp[:, 1] /= 0.1994
		temp[:, 2] /= 0.2010
	else:
		print('New dataset! Update preprocessing!')
		raise('')

	return temp

def postprocess(img, dataset=None, net_type=None):
	"""
	Postprocess by unnormalizing

	Single image only.
	"""
	temp = img.clone()

	if net_type=="inception":
		temp = (temp * 128.0) + 127.5
	
	elif dataset=="multi-pie":		
		temp[:, 0] *= 1.0
		temp[:, 1] *= 1.0
		temp[:, 2] *= 1.0
		temp[:, 0] += 131.0912
		temp[:, 1] += 103.8827
		temp[:, 2] += 91.4953

	elif dataset=="fmnist":
		pass
	elif dataset=="tiny-imagenet":
		temp[:, 0] *= 0.2302
		temp[:, 1] *= 0.2265
		temp[:, 2] *= 0.2262
		temp[:, 0] += 0.4802
		temp[:, 1] += 0.4481
		temp[:, 2] += 0.3975
	elif dataset=="cifar10-trades":
		pass
	elif dataset=='mnist':
		pass
	elif dataset=="cifar10-feat-scat":
		temp[:, 0] *= 0.5
		temp[:, 1] *= 0.5
		temp[:, 2] *= 0.5
		temp[:, 0] += 0.5
		temp[:, 1] += 0.5
		temp[:, 2] += 0.5
	elif dataset=="imagenet":
		temp[:, 0] *= 0.229
		temp[:, 1] *= 0.224
		temp[:, 2] *= 0.225	
		temp[:, 0] += 0.485
		temp[:, 1] += 0.456
		temp[:, 2] += 0.406
	elif dataset=="cifar10":
		temp[:, 0] *= 0.2023
		temp[:, 1] *= 0.1994
		temp[:, 2] *= 0.2010
		temp[:, 0] += 0.4914
		temp[:, 1] += 0.4822
		temp[:, 2] += 0.4465
	else:
		print('New dataset! Update postprocessing!')
		raise('')

	return temp

def margin_loss(logits,y):
	logit_org = logits.gather(1,y.view(-1,1))
	logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
	loss = -logit_org + logit_target
	loss = torch.sum(loss)
	return loss