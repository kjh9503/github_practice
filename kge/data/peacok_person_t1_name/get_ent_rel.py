import pickle

with open('ent2id.pkl', 'rb') as f :
    ent2id = pickle.load(f)
with open('rel2id.pkl', 'rb') as f :
    rel2id = pickle.load(f)

id2ent = {pair[1] : pair[0] for pair in ent2id.items()}
id2rel = {pair[1] : pair[0] for pair in rel2id.items()}

ents = []
rels = []
for i in range(len(id2ent)) :
    line = f"{i}\t{id2ent[i]}\n"
    ents.append(line)

for i in range(len(id2rel)) :
    line = f"{i}\t{id2rel[i]}\n"
    rels.append(line)

with open('entities.dict', 'w') as f :
    f.writelines(ents)
with open('relations.dict', 'w') as f :
    f.writelines(rels)