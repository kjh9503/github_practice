import json
import pickle
import torch
from tqdm import tqdm, trange
from typing import Dict, List, Tuple, Set
from collections import defaultdict, deque
from util import verbalize_subgraph_triples



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

        # Tokenization (B, L)
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
    with open('rel2id.pkl', 'rb') as f :
        rel2id = pickle.load(f)
    # Build adjacency map: entity → set of (r, neighbor)
    adj = defaultdict(set)
    for h, r, t in triples:
        adj[h].add((r, t))

    entity2subgraph = {}

    for ent in tqdm(target_entities):
        visited: Set[str] = set([ent])
        frontier = deque([ent])
        collected_triples = set()

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
            frontier = next_frontier


        
        job_triples = [t for t in list(collected_triples) if (t[1] == rel2id['has a job of'] or t[1] == 'has a job of')]
        assert len(job_triples) > 0, (ent, collected_triples)
        other_triples = [t for t in list(collected_triples) if (t[1] != rel2id['has a job of'] or t[1] != 'has a job of')]

        entity2subgraph[ent] = list(set(job_triples + other_triples))

    return entity2subgraph



def get_ent2subg(data_path = '../dataprocessing/peacok_person_t2_name', code='t2'):
    assert code in data_path
    
    with open(f'{data_path}/person_to_name.json', 'r') as f :
        p2n = json.load(f)
    n2p = {pair[1] : pair[0] for pair in p2n.items()}
    
    with open('ent2id.pkl', 'rb') as f :
        ent2id = pickle.load(f)
    with open('rel2id.pkl', 'rb') as f :
        rel2id = pickle.load(f)
    with open(f'{data_path}/train.txt', 'r') as f :
        train = f.readlines()
    train_ind = []
    for line in train :
        h, r, t = line.strip().split('\t')
        h_i = ent2id[h]
        r_i = rel2id[r]
        t_i = ent2id[t]
        newline = f"{h_i}\t{r_i}\t{t_i}\n"
        train_ind.append(newline)

    

    person_name = p2n
    name_person = {pair[1] : pair[0] for pair in person_name.items()}
    
    target_ents_ind = [ent2id[e] for e in ent2id if e in name_person]
    assert len(target_ents_ind) > 0, len(target_ents_ind)
    target_ents = []
    
    triples = []
    for line in train :
        h, r, t = line.strip().split('\t')
        if h in n2p:
            target_ents.append(h)
        triples.append((h, r, t))

    triples_ind = [tuple(map(int, line.strip().split('\t'))) for line in train_ind]

    ent2sub_ind = extract_entity_subgraphs(triples_ind, target_ents_ind, 2)
    ent2sub = extract_entity_subgraphs(triples, target_ents, 2)
    print('ent2sub_ind : ', len(ent2sub_ind))
    print('ent2sub : ', len(ent2sub))
    
    
    with open(f'./data/ent2sub_{code}.pkl', 'wb') as f :
        pickle.dump(ent2sub, f)
    with open(f'./data/ent2sub_ind_{code}.pkl', 'wb') as f :
        pickle.dump(ent2sub_ind, f)

def get_subg_txt_repr(data_path = '../dataprocessing/peacok_person_t2_name', time='t2', encoder='t1gpt', model_name = '../finetuned_models/gpt-j-6B_t1_chain_name_gptj_full'):
    
    assert time in data_path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

    with open(f"./data/ent2sub_{time}.pkl", 'rb') as f :
        ent2sub = pickle.load(f)
    head2text_repr = compute_head2text_repr_batched(
                        head2subgraph_triples=ent2sub,
                        verbalize_subgraph_triples=verbalize_subgraph_triples,
                        tokenizer=tokenizer,
                        llm_model=model,
                        device='cuda',
                        batch_size=32
                    )

    print(f'subgraph_txt_repr_{time}_{encoder} : ', len(head2text_repr))
    with open(f'./data/subgraph_txt_repr_{time}_llm_{encoder}.pkl', 'wb') as f :
        pickle.dump(head2text_repr, f)

    
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        h_idx = -rl - tl - 1
        r_idx = - tl - 1
        t_idx = -1
        position_ids.append([h_idx, r_idx, t_idx])
    
    position_ids = torch.tensor(position_ids, device=device)  # (B, 3)
    B, L, D = last_hidden.size()
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, 3)
    
    selected_embeds = last_hidden[batch_idx, position_ids]  # (B, 3, D)
    
    return selected_embeds, encodings['input_ids'], position_ids


def get_triple_text_repr(data_path = '../dataprocessing/peacok_person_t2_name', time='t1', encoder='t1gpt', model_name = '../finetuned_models/gpt-j-6B_t1_chain_name_gptj_full'):
    
    assert time in data_path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(f"{data_path}/train.txt", 'r') as f :
        train = f.readlines()
    train_str = [line.strip().split('\t') for line in train]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

    all_selected_embeds = []

    model.eval()
    batch_size = 64
    for i in trange(0, len(train_str), batch_size):
        batch = train_str[i:i+batch_size]
        with torch.no_grad():
            selected_embeds_, input_ids, position_ids = encode_triples_from_llm(batch, tokenizer, model, 'cuda')
            all_selected_embeds.append(selected_embeds_.cpu())
    
    selected_embs = torch.cat(all_selected_embeds, dim=0)
    
    print(f'triple to txt emb {time}_{encoder} : ', selected_embs.size())
    torch.save(selected_embs, f'./data/triples_txt_embed_{time}_llm_{encoder}.pt')



    
if __name__ == '__main__' : 


    get_ent2subg(data_path='../data/peacok_person_t1_name', code='t1')
    get_ent2subg(data_path='../data/peacok_person_t1.2_name', code='t1.2')

    get_subg_txt_repr(data_path='../data/peacok_person_t1_name', time='t1', encoder='t1_chain_gpt')
    get_subg_txt_repr(data_path='../data/peacok_person_t1.2_name', time='t1.2', encoder='t1_chain_gpt')

    get_triple_text_repr(data_path='../data/peacok_person_t1_name', time='t1', encoder='t1_chain_gpt')
    get_triple_text_repr(data_path='../data/peacok_person_t1.2_name', time='t1.2', encoder='t1_chain_gpt')
