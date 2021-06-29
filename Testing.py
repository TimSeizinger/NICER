#from utils import hinge_loss
import torch
import cma
import nevergrad as ng

tensor = torch.randn(4) * 10

target = torch.zeros(4)

ngarray = ng.p.Array(init=tensor.tolist()).set_bounds(-100, 100)

print(ngarray)
print(type(ngarray))


optimizer = ng.optimizers.NGOpt(parametrization=ngarray, budget=100)

loss_func = torch.nn.MSELoss(reduction='mean')

for i in range(optimizer.budget):
    print(i)
    candidate = optimizer.ask()
    candidate_tensor = torch.tensor(candidate.value)
    loss = loss_func(candidate_tensor, target).item()
    optimizer.tell(candidate, loss)
    print(optimizer.provide_recommendation().value)