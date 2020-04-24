import torch
import torch.nn.functional as F
from scipy import signal
import numpy as np


# def smoothen_kernel(net, parameter, rho=0.5):
# 	weights = torch.tensor([[rho, rho, rho],
# 							[rho, 1.0, rho],
# 							[rho, rho, rho]])

# 	weights = weights.view(1, 1, 3, 3).repeat(1, 3, 1, 1)
# 	output = F.conv2d(parameter, weights)
# 	return output

# import torch
# import torch.nn.functional as F
# h, w = 5, 5
# x = torch.ones(1, 3, h, w)
# rho=0.5

# weights = torch.tensor([[rho, rho, rho],
#                         [rho, 1.0, rho],
#                         [rho, rho, rho]])

# weights = weights.view(1, 1, 3, 3)
# state = model.state_dict()
# for param in state:
#   if len(list(state[param].size()))==4 and not('downsample' in param):
#     print(param, state[param].size())
#     for i in range(len(state[param])):
#       for j in range(len(state[param][i])):
        
#         try:
#           temp = torch.unsqueeze(torch.unsqueeze(state[param][i][j], 0), 0)
#           state[param][i][j] = F.conv2d(F.pad(temp, (1, 1, 1, 1), mode='reflect'), weights)
#         except:
#           a, b, _, _ = list(state[param].size())
#           temp = torch.ones(a, b, 3, 3)
#           for i in range(len(state[param])):
#             for j in range(len(state[param][i])):
#               temp[i][j] = temp[i][j] * state[param][i][j]

#           state[param] = temp
#           temp = torch.unsqueeze(torch.unsqueeze(state[param][i][j], 0), 0)
#           state[param][i][j] = F.conv2d(F.pad(temp, (1, 1, 1, 1), mode='reflect'), weights)


def generateSmoothKernel(data, r):
	result = np.zeros_like(data)
	[k1, k2, m, n] = data.shape
	mask = np.zeros([3,3])
	for i in range(3):
		for j in range(3):
			if i == 1 and j == 1:
				mask[i,j] = 1
			else:
				mask[i,j] = r
	mask = mask
	for i in range(m):
		for j in range(n):
			result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
	return result