import torch

x = torch.load('personal_circuit.pt')
for i, p in enumerate(x) :
    print(i, p)