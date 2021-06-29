import sklearn
import torch
import numpy as np

param_grid = [{'optim_lr': [0.15, 0.20, 0.25, 0.30, 0.35, 0.40], 'gamma': [0.05, 0.10, 0.15, 0.20], 'score_pow': [1.0, 1.5, 2.0, 2.5, 3.0], 'composite_balance': [-0.5, -0.25, 0, 0.25, 0.5]}]

tensor = torch.zeros(3, 244, 244)
tensor_np = tensor.numpy()

print(tensor_np.shape)

tensor_np = np.transpose(tensor_np)

print(tensor_np.shape)