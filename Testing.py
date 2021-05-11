import torch
from utils import weighted_mean

score = torch.tensor([[0.0190, 0.0295, 0.0620, 0.1534, 0.2769, 0.2328, 0.1208, 0.0585, 0.0290, 0.0179]])
weights = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
print(score)
print(weights)

w_mean = weighted_mean(score, weights)
print(w_mean)