import torch

x = torch.load('checkpoint')
model = x['model_state_dict']
ent = model['entity_embedding']
rel = model['relation_embedding']

torch.save(ent, '../../../align/data/entity_embedding_t1.pt')
torch.save(rel, '../../../align/data/relation_embedding_t1.pt')