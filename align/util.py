
import re
import dgl
import torch
import pickle
from typing import List, Dict
import random
import json
from collections import defaultdict
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text

def is_correct(pred, target):
    norm_pred = normalize_text(pred)
    norm_target = normalize_text(target)
    if norm_target in norm_pred:
        return True
    else: 
        return False

def load_gnn_config():
    with open('ent2id.pkl', 'rb') as f :
        ent2id = pickle.load(f)
    with open('rel2id.pkl', 'rb') as f :
        rel2id = pickle.load(f)
    
    gnn_dim_factor = 1
    num_nodes = len(ent2id)
    num_rels = len(rel2id)
    h_dim = 4096
    out_dim = h_dim
    num_bases = 100
    num_basis = 100
    num_hidden_layers = 2
    dropout = 0.15
    self_loop = True
    skip_connect = False
    encoder_name = 'uvrgcn'
    opn = 'sub'
    use_cuda = True
    analysis = False
    rel_emb = None
    return num_nodes, h_dim, out_dim, num_rels, num_bases, num_basis, num_hidden_layers, dropout, self_loop, skip_connect, encoder_name, opn, rel_emb, use_cuda, analysis



def load_graph(triples, ent2id, rel2id, h_dim):
    
    
  # triples: List of (head, relation, tail) index tuples
  heads, rels, tails = zip(*triples)

  # DGL의 heterograph를 만들되, relation은 edge_type으로 넘김
  g = dgl.graph((heads, tails), num_nodes=len(ent2id)).to('cuda')  # num_ents는 전체 entity 수

  rel_tensor = torch.tensor(rels, dtype=torch.long).to('cuda')  # (num_edges,)

  # edge type (relation) 저장
  g.edata['rel_type'] = rel_tensor

  init_ent_emb = torch.nn.Embedding(len(ent2id), h_dim).weight.cuda()  # shape: (num_ents, h_dim)
  init_rel_emb = torch.nn.Embedding(len(rel2id), h_dim).weight.cuda() # shape: (num_rels, h_dim)

  g.ndata['feat'] = init_ent_emb
  g.edata['init_rel_emb'] = init_rel_emb[rel_tensor]  # ← 이 부분 추가

  deg = g.in_degrees().float().clamp(min=1)
  g.ndata['norm'] = torch.pow(deg, -0.5).unsqueeze(1).to(init_ent_emb.device)

  # forward
  return g, init_ent_emb, init_rel_emb



def verbalize_subgraph_triples(triples):
    """
    subgraph_triples: List of list of (h, r, t) tuples in string form, e.g.,
      [
        [('kim', 'has a job of', 'pilot'), ('kim', 'lives in', 'seoul')],
        [('john', 'knows', 'mary')],
        ...
      ]
    returns:
      [
        "kim has a job of pilot. kim lives in seoul.",
        "john knows mary.",
        ...
      ]
    """
    verbalized = ''
    for triple in triples:
        h, r, t = triple
        sent = "{} {} {}.".format(h, r, t)
        verbalized = verbalized + sent + ' '
    verbalized = verbalized.strip()
    return verbalized


rel2prompt = {'is a social characteristic of' : 'is a social characteristic of', 
              'characteristic' : 'has a characteristic of',
              'characteristic_relationship' : 'has a social characteristic of',
              'experience' : 'has an experience of',
              'experience_relationship' : 'has a social experience of',
              'goal_plan' : 'has a goal or plan of',
              'goal_plan_relationship' : 'has a social goal or plan of',
              'is a characteristic of' : 'is a characteristic of',
              'is a goal or plan of' : 'is a goal or plan of',
              'is a routine or habit of' : 'is a routine or habit of',
              'is a social experience of' : 'is a social experience of',
              'is a social goal or plan of' : 'is a social goal or plan of',
              'is a social routine or habit of' : 'is a social routine or habit of',
              'is an experience of' : 'is an experience of',
              'routine_habit' : 'has a routine or habit of',
              'routine_habit_relationship' : 'has a social routine or habit of',
              'has a job of' : 'has a job of'
              }
ent2beforenorm = {'scaled Mt Everest' : 'scaled Mt. Everest', 
                  'to invest in philanthropic causes' : 'to invest in philanthropic causes.',
                  'to hike to the top of Mt Everest' : 'to hike to the top of Mt. Everest',
                  'to create a personal sanctuary of peace and serenity' : 'to create a personal sanctuary of peace and serenity.',
                  'completed a background check for babysitting' : 'completed a background check for babysitting.',
                  'to score a winning touchdown in a championship game' : 'to score a winning touchdown in a championship game.',
                  'to achieve a healthy weight' : 'to achieve a healthy weight.',
                  'finished a 262 mile race' : 'finished a 26.2 mile race',
                  'completes tasks quickly and efficiently' : 'completes tasks quickly and efficiently.',
                  'to help my clients achieve their fitness goals' : 'to help my clients achieve their fitness goals.'}

def retrieve_loc_triple(loc_prompt, t1_hr2t, ent2ind, rel2ind, t1_p2j, t2_p2j, sd=42) :
    #prompt1 : famous person has a characteristic of -> popular
    #prompt2 : popular is a social characteristic of -> quarterback
    #hr2t : t1 kg
    random.seed(sd)
    t1_j2p = {pair[1] : pair[0] for pair in t1_p2j.items()}

    jobs = list(t1_p2j.values())

    sorted_rels = sorted(rel2ind.keys(), key=lambda x: -len(x))
    for rel in sorted_rels:
      if rel == 'is a job of' : continue
      if rel2prompt[rel] not in loc_prompt:
         continue
      head = loc_prompt.split(rel2prompt[rel])[0].strip()
      if head in ent2beforenorm :
        head = ent2beforenorm[head]
      
      assert head in ent2ind, head
      tails = t1_hr2t[(head, rel)]
      #prompt1 : head : famous person, rel : characteristic, tails = {popular}
      #prompt2 : head : popular, rel : characteristic_relationship, tails= {quarterback}
      if head in jobs : #case 1 : head 가 job -> job의 t1 person으로 triple 만들기
          t1_job = head
          t1_person = t1_j2p[head]
          t1_triple = (t1_person, 'has a job of', t1_job)
          t2_job = t2_p2j[t1_person]
          t2_triple = (t1_person, 'has a job of', t2_job)

          
      elif list(tails)[0] in jobs : #case 2 : quarter back 의 t1 person 갖고오고, t1 person 의 t2 job 갖고오기
        t1_rand_job = random.choice(list(tails))
        t1_person = t1_j2p[t1_rand_job]
        t1_triple = (t1_person, 'has a job of', t1_rand_job)
        t2_job = t2_p2j[t1_person]
        t2_triple = (t1_person, 'has a job of', t2_job)
      
      else :
        raise ValueError('Invalid Case : ', loc_prompt)
      return (t2_triple, t1_triple)    
    
if __name__ == '__main__' :
  with open('data/loc_t2/locality.json', 'r') as f :
    loc = json.load(f)
  with open('../dataprocessing/peacok_person_t1_name/train.txt', 'r') as f :
    t1_lines = f.readlines()
  with open('../dataprocessing/peacok_person_t2_name/train.txt', 'r') as f :
    t2_lines = f.readlines()
  with open('../dataprocessing/peacok_person_t1_name/person_to_name.json', 'r') as f:
    p2n = json.load(f)
  with open('ent2id.pkl', 'rb') as f :
    ent2id = pickle.load(f)
  with open('rel2id.pkl', 'rb') as f :
    rel2id = pickle.load(f)
  names = list(p2n.values())

  t1_hr2t = defaultdict(set)
  t1_p2j = {}
  t2_p2j = {}
  for line in t1_lines :
    h, r, t = line.strip().split('\t')
    t1_hr2t[(h,r)].add(t)
    if h in names and r == 'has a job of' :
        t1_p2j[h] = t
  assert len(t1_p2j) > 0
  for line in t2_lines :
    h, r, t = line.strip().split('\t')
    if h in names and r == 'has a job of' :
        t2_p2j[h] = t
  assert len(t2_p2j) > 0

  loc_prompts = [d['prompt'] for d in loc]
  loc_answers = [d['answer'] for d in loc]


  for i in (random.sample([i for i in range(len(loc_prompts))], 10)): 
    triples = retrieve_loc_triple(loc_prompts[i], t1_hr2t, ent2id, rel2id, t1_p2j, t2_p2j)
    print('loc data : ', loc_prompts[i], loc_answers[i])
    print(triples)
    print('='*80)