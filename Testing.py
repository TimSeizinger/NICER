#from utils import hinge_loss
import torch
import cma

tensor = torch.randn(4)

target = torch.zeros(4)

optimizer = cma.CMAEvolutionStrategy(x0=tensor.tolist(), sigma0=1, inopts={'popsize': 5, 'bounds': [-100, 100]})

loss_func = torch.nn.MSELoss(reduction='mean')

candi = [100,100,100,1000]
candi = [candi/100 for candi in candi]

print(candi)

for i in range(20):
    candidates = optimizer.ask()
    losses = []
    for candidate in candidates:
        candidate = torch.tensor(candidate)
        losses.append(loss_func(candidate, target).item())
    optimizer.tell(candidates, losses)
    print(len(losses))
    print(optimizer.best.x)
    if i == 10:
        optimizer.opts.set({'popsize': 10})
