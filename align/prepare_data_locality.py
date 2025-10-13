import json
import pickle
import torch
from tqdm import tqdm, trange
from typing import Dict, List, Tuple, Set
from collections import defaultdict, deque
# from util import verbalize_subgraph_triples
import torch
with open('../data/peacok_person_t1_name/person_to_name.json', 'r') as f :
    p2n = json.load(f)
n2p = {pair[1] : pair[0] for pair in p2n.items()}

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
        if h in ent2beforenorm :
            h = ent2beforenorm[h]
        if r in rel2prompt :
            r = rel2prompt[r]
        if t in ent2beforenorm :
            t = ent2beforenorm

        sent = "{} {} {}.".format(h, r, t)
        verbalized = verbalized + sent + ' '
    verbalized = verbalized.strip()
    return verbalized

def recover_triple_from_prompt(prompt, answer, ent2ind, rel2ind):
    sorted_rels = sorted(rel2ind.keys(), key=lambda x: -len(x))
    for rel in sorted_rels:
        # if 'job' in rel :
        #     continue
        if 'is a job of' in rel :
            continue
        if rel2prompt[rel] in prompt:
            head = prompt.split(rel2prompt[rel])[0].strip()
            if head in ent2beforenorm :
                head = ent2beforenorm[head]
            if answer in ent2beforenorm :
                answer = ent2beforenorm[answer]
            assert head in ent2ind, head
            assert answer in ent2ind, answer
            return (head, rel, answer)
    print(prompt)
        


def compute_head2text_repr_batched(
    head2subgraph_triples: Dict[str, List[Tuple[str, str, str]]],
    verbalize_subgraph_triples: callable,
    tokenizer,
    llm_model,
    device: str = 'cuda',
    batch_size: int = 16,
    max_length: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Batched version of subgraph textual encoding.
    """
    head2text_repr = {}
    heads = list(head2subgraph_triples.keys())

    for i in tqdm(range(0, len(heads), batch_size), desc="Batch Encoding Subgraphs"):
        batch_heads = heads[i:i + batch_size]
        batch_triples = [head2subgraph_triples[h] for h in batch_heads]
        batch_texts = [verbalize_subgraph_triples(triples) for triples in batch_triples]


        tokenized = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = llm_model(**tokenized, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.hidden_states[-1]  # (B, L, D)
            attention_mask = tokenized["attention_mask"].unsqueeze(-1)  # (B, L, 1)
            summed = (last_hidden * attention_mask).sum(1)              # (B, D)
            count = attention_mask.sum(1)                               # (B, 1)
            mean_pooled = summed / count                                # (B, D)

        for h, repr in zip(batch_heads, mean_pooled):
            head2text_repr[h] = repr.cpu()

    return head2text_repr


def extract_entity_subgraphs(
    triples: List[Tuple[str, str, str]],
    target_entities: List[str],
    id2ent,
    id2rel, 
    n_hop: int = 2,
    
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Extract n-hop subgraphs around given entities from a list of KG triples.

    Args:
        triples: List of all triples in KG (h, r, t)
        target_entities: List of head entities to extract subgraphs for
        n_hop: Number of hops to expand (default=2)

    Returns:
        Dict of {entity: List of (h, r, t) in its n-hop subgraph}
    """
    
    # Build adjacency map: entity → set of (r, neighbor)
    adj = defaultdict(set)
    for h, r, t in triples:
        adj[h].add((r, t))

    entity2subgraph = {}
    entity2subgraph_txt = {}
    
    for ent in tqdm(target_entities):
        visited: Set[str] = set([ent])
        frontier = deque([ent])
        collected_triples = set()
        collected_triples_text = set()

        for _ in range(n_hop):
            next_frontier = deque()
            while frontier:
                curr = frontier.popleft()
                for r, neighbor in adj.get(curr, []):

                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                    # Always add (curr, r, neighbor) if present in KG
                    if (curr, r, neighbor) in triples:
                        collected_triples.add((curr, r, neighbor))
                        collected_triples_text.add((id2ent[curr], id2rel[r], id2ent[neighbor]))
            frontier = next_frontier

        entity2subgraph[ent] = list(collected_triples)
        entity2subgraph_txt[id2ent[ent]] = list(collected_triples_text)

    return entity2subgraph, entity2subgraph_txt



def get_ent2subg(time='t1.2', n_hop=2):
    with open('ent2id.pkl', 'rb') as f :
        ent2id = pickle.load(f)
    with open('rel2id.pkl', 'rb') as f :
        rel2id = pickle.load(f)
        
    id2ent = {pair[1] : pair[0] for pair in ent2id.items()}
    id2rel = {pair[1] : pair[0] for pair in rel2id.items()}
    
    loc_fname = f'./data/loc_{time}/locality_triple_inds_{time}.json'
    kg_fname = f'../data/peacok_person_{time}_name/train.txt'

    ent2sub_fname = f'./data/loc_{time}/ent2sub_loc_{time}.pkl'
    ent2sub_ind_fname = f'./data/loc_{time}/ent2sub_ind_loc_{time}.pkl'


    with open(loc_fname, 'r') as f:
        loc_triple_inds = json.load(f)
    with open(kg_fname, 'r') as f :
        train = f.readlines()

    triples_ind = []
    triples_txt = []
    for line in train :
        h, r, t = line.strip().split('\t')
        h_i = ent2id[h]
        r_i = rel2id[r]
        t_i = ent2id[t]
        triples_ind.append((h_i, r_i, t_i))
        triples_txt.append((h, r, t))
        

    loc_head_inds = [triple[0] for triple in loc_triple_inds]
    loc_head_txts = [id2ent[triple[0]] for triple in loc_triple_inds]


    ent2sub_ind, ent2sub_txt = extract_entity_subgraphs(triples_ind, loc_head_inds, id2ent, id2rel, n_hop)
    print('ent2sub_ind : ', len(ent2sub_ind))
    

    with open(ent2sub_ind_fname, 'wb') as f :
        pickle.dump(ent2sub_ind, f)
    with open(ent2sub_fname, 'wb') as f :
        pickle.dump(ent2sub_txt, f)


def get_subg_txt_repr(time='t1.2', encoder='t1_chain_gpt'):
    
    ent2sub_fname = f'./data/loc_{time}/ent2sub_loc_{time}.pkl'
    save_fname = f'./data/loc_{time}/subgraph_txt_repr_{time}_llm_{encoder}.pkl'
    
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = '../finetuned_models/gpt-j-6B_t1_chain_name_gptj_full'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
    

    with open(ent2sub_fname, 'rb') as f :
        ent2sub = pickle.load(f)
        
    head2text_repr = compute_head2text_repr_batched(
                        head2subgraph_triples=ent2sub,
                        verbalize_subgraph_triples=verbalize_subgraph_triples,
                        tokenizer=tokenizer,
                        llm_model=model,
                        device='cuda',
                        batch_size=16
                    )

    print(f'subgraph_txt_repr_{time}_{encoder} : ', len(head2text_repr))
    with open(save_fname, 'wb') as f :
        pickle.dump(head2text_repr, f)

    

@torch.no_grad()
def encode_triples_from_llm(triples, tokenizer, model, device='cuda'):
    """
    Args:
        triples: List of (head, relation, tail) strings
        tokenizer: HF tokenizer
        model: HF LLM (e.g., GPT-J)
        device: 'cuda' or 'cpu'

    Returns:
        selected_embeds: Tensor of shape (B, 3, D)
        input_ids: Tensor of shape (B, L)
        position_ids: Tensor of shape (B, 3)
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    heads, rels, tails = zip(*triples)

    # 1. 각 구성요소 token length 계산
    head_tok = tokenizer(list(heads), add_special_tokens=False)['input_ids']
    rel_tok = tokenizer([' ' + r for r in list(rels)], add_special_tokens=False)['input_ids']
    tail_tok = tokenizer([' ' + t for t in list(tails)], add_special_tokens=False)['input_ids']

    head_lens = [len(x) for x in head_tok]
    rel_lens = [len(x) for x in rel_tok]
    tail_lens = [len(x) for x in tail_tok]

    # 2. 전체 문장 구성
    full_texts = [f"{h} {r} {t}" for h, r, t in triples]
    encodings = tokenizer(
        full_texts,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)

    outputs = model(**encodings, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]  # (B, L, D)

    # 3. position 계산
    position_ids = []
    
    for hl, rl, tl in zip(head_lens, rel_lens, tail_lens):
        h_idx = hl - 1
        r_idx = hl + rl - 1
        t_idx = hl + rl + tl - 1
        position_ids.append([h_idx, r_idx, t_idx])
    
    position_ids = torch.tensor(position_ids, device=device)  # (B, 3)
    B, L, D = last_hidden.size()
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, 3)
    gather_idx = torch.clamp(position_ids, max=L - 2)

    selected_embeds = last_hidden[batch_idx, gather_idx]  # (B, 3, D)
    return selected_embeds, encodings['input_ids'], position_ids


def get_triple_text_repr(triple_txt, time='t2', encoder='t1gpt'):
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    
    model_name = '../finetuned_models/gpt-j-6B_t1_chain_name_gptj_full'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

    all_selected_embeds = []

    model.eval()
    batch_size = 16
    for i in trange(0, len(triple_txt), batch_size):
        batch = triple_txt[i:i+batch_size]
        with torch.no_grad():
            selected_embeds_, input_ids, position_ids = encode_triples_from_llm(batch, tokenizer, model, 'cuda')
        
        all_selected_embeds.append(selected_embeds_.cpu())
    
    selected_embs = torch.cat(all_selected_embeds, dim=0)
    
    print(f'triple to txt emb {encoder} : ', selected_embs.size())
    torch.save(selected_embs, f'./data/loc_{time}/triples_txt_embed_loc_{time}_llm_{encoder}.pt')


    
if __name__ == '__main__' : 
    
    
    with open('./data/loc_t1.2/locality_t1.2.json', 'r') as f :
        locality_t12 = json.load(f)
    with open('ent2id.pkl', 'rb') as f :
        ent2id = pickle.load(f)
    with open('rel2id.pkl', 'rb') as f :
        rel2id = pickle.load(f)
        
    
    # get_ent2subg(time='t1.2')

    # get_subg_txt_repr(time = 't1.2', encoder='t1_chain_gpt')

        
    

    locality_triple_txts_t12 = []


    
    for d in locality_t12 :
        prompt = d['prompt']
        answer = d['answer']
        triple_txt = recover_triple_from_prompt(prompt, answer, ent2id, rel2id)
        locality_triple_txts_t12.append(triple_txt)

    get_triple_text_repr(triple_txt=locality_triple_txts_t12, time='t1.2', encoder='t1gpt_chain')

    
    
