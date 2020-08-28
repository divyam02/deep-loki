from sewar.full_ref import uqi

import numpy as np

def get_uiqi(img_batch, pert_batch):
	score = 0.0

	for i in range(len(img_batch)):
		og_img = img_batch[i].data.cpu().permute(1, 2, 0).numpy() * 255.0
		pert_img = pert_batch[i].data.cpu().permute(1, 2, 0).numpy() * 255.0
		score += uqi(og_img, pert_img)

	return score