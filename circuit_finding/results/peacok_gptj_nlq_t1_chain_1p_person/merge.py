import torch
import json
from tqdm import tqdm, trange
with open('person2id.json', 'r') as f:
    p2i = json.load(f)

i2p = {pair[1] : pair[0] for pair in p2i.items()}

personal_circuit = {}
for i in trange(822):
    person = i2p[i]
    circ = torch.load(f'info_{i}.pt')
    personal_circuit[person] = {'path' : circ['path']}
    
torch.save(personal_circuit, 'personal_circuit.pt')

