"""

Using https://github.com/nitarshan/one-pixel-attack. Thanks!

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

CIFAR_LABELS = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def show(img, label):
	og_img = img.unsqueeze(0).cpu().numpy()

	og_img[0][0] *= 0.2023
	og_img[0][1] *= 0.1994
	og_img[0][2] *= 0.2010
	og_img[0][0] += 0.4914
	og_img[0][1] += 0.4822
	og_img[0][2] += 0.4465

	plt.imshow(np.transpose(og_img[0], (1,2,0)), interpolation='nearest')
	plt.savefig('img'+label+'.png')

def tell(img, label, model, target_label=None):
	print("True Label:", CIFAR_LABELS[label], label)
	print("Prediction:", CIFAR_LABELS[model(img.unsqueeze(0)).max(-1)[1]], model(img.unsqueeze(0)).max(-1)[1][0].item())
	print("Label Probabilities:", F.softmax(model(img.unsqueeze(0)).squeeze(), dim=0))
	print("True Label Probability:", F.softmax(model(img.unsqueeze(0)).squeeze(), dim=0)[label].item())
	if target_label is not None:
		print("Target Label Probability:", F.softmax(model(img.unsqueeze(0)).squeeze(), dim=0)[target_label].item())

def perturb(p, img):
	# Elements of p should be in range [0,1]
	img_size = img.size(1) # C x _H_ x W, assume H == W
	p_img = img.clone()
	xy = (p[0:2].copy() * img_size).astype(int)
	xy = np.clip(xy, 0, img_size-1)
	rgb = p[2:5].copy()
	rgb = np.clip(rgb, 0, 1)
	p_img[:,xy[0],xy[1]] = torch.from_numpy(rgb)
	return p_img

def visualize_perturbation(p, img, label, model, target_label=None):
	p_img = perturb(p, img)
	print("Perturbation:", p)
	show(p_img, str(label.data))
	tell(p_img, label, model, target_label)
	np.save('adv_'+str(label.data.cpu()), p_img.unsqueeze(0).data.cpu().numpy())

def evaluate(candidates, img, label, model):
	preds = []
	model.eval()
	with torch.no_grad():
		for i, xs in enumerate(candidates):
			p_img = perturb(xs, img)
			preds.append(F.softmax(model(p_img.unsqueeze(0)).squeeze(), dim=0)[label].item())
	return np.array(preds)

def evolve(candidates, F=0.5, strategy="clip"):
	gen2 = candidates.copy()
	num_candidates = len(candidates)
	for i in range(num_candidates):
		x1, x2, x3 = candidates[np.random.choice(num_candidates, 3, replace=False)]
		x_next = (x1 + F*(x2 - x3))
		if strategy == "clip":
			gen2[i] = np.clip(x_next, 0, 1)
		elif strategy == "resample":
			x_oob = np.logical_or((x_next < 0), (1 < x_next))
			x_next[x_oob] = np.random.random(5)[x_oob]
			gen2[i] = x_next
	return gen2

def attack(model, img, true_label, target_label=None, iters=1000, pop_size=400, verbose=True):
	# Targeted: maximize target_label if given (early stop > 50%)
	# Untargeted: minimize true_label otherwise (early stop < 5%)
	candidates = np.random.random((pop_size,5))
	candidates[:,2:5] = np.clip(np.random.normal(0.5, 0.5, (pop_size, 3)), 0, 1)
	is_targeted = target_label is not None
	label = target_label if is_targeted else true_label
	fitness = evaluate(candidates, img, label, model)

	def is_success():
		return (is_targeted and fitness.max() > 0.5) or ((not is_targeted) and fitness.min() < 0.05)

	for iteration in range(iters):
		# Early Stopping
		if is_success():
			break
		if verbose and iteration%100 == 0: # Print progress
			print("Target Probability [Iteration {}]:".format(iteration), fitness.max() if is_targeted else fitness.min())
		# Generate new candidate solutions
		new_gen_candidates = evolve(candidates, strategy="resample")
		# Evaluate new solutions
		new_gen_fitness = evaluate(new_gen_candidates, img, label, model)
		# Replace old solutions with new ones where they are better
		successors = new_gen_fitness > fitness if is_targeted else new_gen_fitness < fitness
		candidates[successors] = new_gen_candidates[successors]
		fitness[successors] = new_gen_fitness[successors]
	best_idx = fitness.argmax() if is_targeted else fitness.argmin()
	best_solution = candidates[best_idx]
	best_score = fitness[best_idx]
	if verbose:
		visualize_perturbation(best_solution, img, true_label, model, target_label)
	return is_success(), best_solution, best_score