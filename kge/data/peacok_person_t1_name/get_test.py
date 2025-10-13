import json
with open('train.txt', 'r') as f :
    lines = f.readlines()
with open('person_to_name.json' ,'r') as f :
    p2n = json.load(f)
names = list(p2n.values())

test = []
for line in lines :
    h, r, t = line.strip().split('\t')
    if h in names or t in names :
        test.append(line)

print(len(test))
with open('test.txt', 'w') as f :
    f.writelines(test)