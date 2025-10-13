import dgl
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn import GNN
from typing import List
from util import load_graph, load_gnn_config, verbalize_subgraph_triples
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_head_id2graph(headid2subgraph: dict, num_ents: int, num_rels: int):
    """
    headid2subgraph: Dict[int, List[List[int]]], subgraph triples per head entity
    num_ents: number of entities in KG
    num_rels: number of relations in KG

    Returns:
        head_id2graph: Dict[int, DGLGraph]
    """
    head_id2graph = {}

    for head_id, triple_ids in headid2subgraph.items():
        h_list, r_list, t_list = zip(*triple_ids)  # list of int

        # Collect all entity IDs
        entity_ids = list(set(h_list + t_list))
        ent_id_to_local = {eid: i for i, eid in enumerate(entity_ids)}  # mapping to local node idx

        # Build edge list (local node indices)
        src = [ent_id_to_local[h] for h in h_list]
        dst = [ent_id_to_local[t] for t in t_list]
        rel = list(r_list)

        g = dgl.graph((src, dst), num_nodes=len(entity_ids))
        g.edata['rel_type'] = torch.tensor(rel, dtype=torch.long)  # (E,)

        # node ID 저장 (global entity ID)
        g.ndata['id'] = torch.tensor(entity_ids, dtype=torch.long)  # (N,)
        g.ndata['feat'] = g.ndata['id']  # 초기 feature용 (embedding lookup index로 사용)

        # head node의 local index 지정
        head_local = ent_id_to_local[head_id]
        g.ndata['head_nid'] = torch.tensor([head_local], dtype=torch.long)

        head_id2graph[head_id] = g

    return head_id2graph


class TextEncoderWithTripleExtract:
    def __init__(self, tokenizer, model, device='cuda'):
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_triples(self, triples):
        """
        triples: List of (head, relation, tail) strings
        Returns:
            Tensor of shape (B, 3, D): [head_embed, rel_embed, tail_embed]
        """
        heads, rels, tails = zip(*triples)

        # 1. 각 구성요소 token length 계산
        head_tok = self.tokenizer(list(heads), add_special_tokens=False)['input_ids']
        rel_tok = self.tokenizer(list(rels), add_special_tokens=False)['input_ids']
        tail_tok = self.tokenizer(list(tails), add_special_tokens=False)['input_ids']

        head_lens = [len(x) for x in head_tok]
        rel_lens = [len(x) for x in rel_tok]
        tail_lens = [len(x) for x in tail_tok]

        # 2. 전체 문장 구성
        full_texts = [f"{h} {r} {t}" for h, r, t in triples]
        encodings = self.tokenizer(
            full_texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encodings, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

        # 3. position 계산
        position_ids = []
        for hl, rl, tl in zip(head_lens, rel_lens, tail_lens):
            h_idx = hl - 1
            r_idx = hl + rl - 1
            t_idx = hl + rl + tl - 1
            position_ids.append([h_idx, r_idx, t_idx])
        
        position_ids = torch.tensor(position_ids, device=self.device)  # (B, 3)

        # 4. Indexing with gather
        B, L, D = hidden_states.size()
        batch_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, 3)  # (B, 3)
        gather_idx = torch.clamp(position_ids, max=L - 2)

        selected_embeds = hidden_states[batch_idx, gather_idx]  # (B, 3, D)
        return selected_embeds, encodings['input_ids'], position_ids



class TripleTextAligner(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        # output projection: dim → out_dim
        self.proj = nn.Linear(dim, out_dim) if dim != out_dim else nn.Identity()
    
    def forward(self, text_embed, triple_embed, text_mask=None, triple_mask=None):
        attn_out, _ = self.cross_attn(query=text_embed, key=triple_embed, value=triple_embed, key_padding_mask=triple_mask)
        x = self.norm(text_embed + attn_out)
        x = x + self.ff(x)
        x = self.proj(x)  # project to out_dim
        return x
    
class PromptTextAligner(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, text_embed, prompt, text_mask=None, triple_mask=None):
        attn_out, _ = self.cross_attn(query=prompt, key=text_embed, value=text_embed, key_padding_mask=triple_mask)
        x = self.norm(text_embed + attn_out)
        x = x + self.ff(x)
        return x



class AlignModule_Lite(nn.Module):
    def __init__(self, ent2id, rel2id, num_heads=8, kge_dim=4096):
        super().__init__()


        # self.triple_text_encoder = TextEncoderWithTripleExtract(self.tokenizer, text_encoder)

        self.gnn_encoder = self._load_gnn_encoder(ent2id, rel2id)


        self.aligner = TripleTextAligner(dim=kge_dim, out_dim=kge_dim, num_heads=num_heads)



    def _load_gnn_encoder(self, ent2id, rel2id):
        # 이미 학습된 gnn encoder를 로드
        gnn_config = load_gnn_config()
        gnn = GNN(*gnn_config).cuda()
        return gnn


    def encode_kge_triple(self, triples_idx, ent_embed, rel_embed):  # triples_idx: Tensor of shape (B, 3)
        """
        triples_idx: torch.LongTensor of shape (B, 3) — (h_id, r_id, t_id)
        returns: (B, 3, D)
        """
        device = next(self.parameters()).device

        h_idx = triples_idx[:, 0]
        r_idx = triples_idx[:, 1]
        t_idx = triples_idx[:, 2]

        h_embed = ent_embed[h_idx] # (B, D)
        r_embed = rel_embed[r_idx]  # (B, D)
        t_embed = ent_embed[t_idx]  # (B, D)

        return torch.stack([h_embed, r_embed, t_embed], dim=1)  # (B, 3, D)

    

    def encode_gnn_subgraphs(self, head_ids, head_id2graph, ent_embed, rel_embed):
        device = next(self.parameters()).device
        subgraph_list = []
        head_nids = []

        for head_id in head_ids:
            triples = head_id2graph[int(head_id)]  # [(h, r, t), ...]

            # 1. Triple → DGLGraph
            graph, head_nid = self.build_dgl_graph_from_triples(triples, head_id)
            subgraph_list.append(graph)
            head_nids.append(head_nid)

        # 2. 마스크 설정
        for g, h_nid in zip(subgraph_list, head_nids):
            g.ndata['head'] = torch.zeros(g.num_nodes(), dtype=torch.bool)
            g.ndata['head'][h_nid] = True
        # 각 subgraph에 초기 임베딩 추가
        # 각 subgraph에 초기 임베딩 추가

        for i, g in enumerate(subgraph_list):
            g = g.to(device)
            g.ndata['id'] = g.ndata['id'].to(device)
            g.ndata['feat'] = ent_embed[g.ndata['id']]
            deg = g.in_degrees().float().clamp(min=1)
            g.ndata['norm'] = torch.pow(deg, -0.5).unsqueeze(1).to(g.device)
            # 'etype'이 없으면 추가해주기
            if 'etype' not in g.edata:
                g.edata['etype'] = g.edata['rel_type'].to(device)

            subgraph_list[i] = g

        batched_graph = dgl.batch(subgraph_list).to(device)

        batched_graph.ndata['h'] = ent_embed[batched_graph.ndata['id']]

        all_node_repr = self.gnn_encoder(batched_graph, ent_embed, rel_embed)
        head_repr = all_node_repr[batched_graph.ndata['head']]  # (B, D)

        return head_repr

    def build_dgl_graph_from_triples(self, triples, head_id):
        """
        triples: List of (h, r, t) in int
        head_id: int
        Returns: (DGLGraph, head_node_index)
        """
        nodes = set()
        for h, _, t in triples:
            nodes.add(h)
            nodes.add(t)

        id_map = {nid: i for i, nid in enumerate(sorted(nodes))}

        src = [id_map[h] for h, _, _ in triples]
        dst = [id_map[t] for _, _, t in triples]
        etype = [r for _, r, _ in triples]

        g = dgl.graph((src, dst), num_nodes=len(id_map))
        g.edata['rel_type'] = torch.tensor(etype, dtype=torch.long)

        # node 'id' feature (original id → for embedding)
        id_tensor = torch.tensor([nid for nid in sorted(nodes)], dtype=torch.long)
        g.ndata['id'] = id_tensor

        head_nid = id_map[int(head_id)]
        return g, head_nid



    def forward(self, text_embed, triples_idx, head_subg_txt_repr, head_id2graph, ent_embed, rel_embed):
        """
        triples_str: list of (h, r, t) in string
        triples_idx: list of (h_id, r_id, t_id) in integer
        """

        # device = next(self.parameters()).device
        # ------------------ Textual Representation -------------------------

        text_embed = torch.cat([text_embed, head_subg_txt_repr.unsqueeze(1)], dim=1) # (B, 4, D)

        # ------------------- Graph Representation -----------------------
        triple_embed = self.encode_kge_triple(triples_idx, ent_embed, rel_embed)
        
        head_ids = triples_idx[:, 0]
        gnn_head_repr = self.encode_gnn_subgraphs(head_ids, head_id2graph, ent_embed, rel_embed).unsqueeze(1) #(B, 1, D)
        triple_embed = torch.cat([triple_embed, gnn_head_repr], dim=1)          # (B, 4, D)

        # === Alignment ===
        aligned = self.aligner(text_embed, triple_embed)
        return aligned  # (B, 4, D)   


class AlignModule_Lite_Circuit(nn.Module):
    def __init__(self, ent2id, rel2id, num_heads=8, kge_dim=4096, lm_num_head=16, num_circuit_head=3, num_layer=28):
        super().__init__()


        # self.triple_text_encoder = TextEncoderWithTripleExtract(self.tokenizer, text_encoder)

        self.gnn_encoder = self._load_gnn_encoder(ent2id, rel2id)

        self.out_dim = int(kge_dim / lm_num_head * num_circuit_head * num_layer)

        self.aligner = TripleTextAligner(dim=kge_dim, out_dim=self.out_dim, num_heads=num_heads)

        

    def _load_gnn_encoder(self, ent2id, rel2id):
        # 이미 학습된 gnn encoder를 로드
        gnn_config = load_gnn_config()
        gnn = GNN(*gnn_config).cuda()
        return gnn


    def encode_kge_triple(self, triples_idx, ent_embed, rel_embed):  # triples_idx: Tensor of shape (B, 3)
        """
        triples_idx: torch.LongTensor of shape (B, 3) — (h_id, r_id, t_id)
        returns: (B, 3, D)
        """
        # device = next(self.parameters()).device

        h_idx = triples_idx[:, 0]
        r_idx = triples_idx[:, 1]
        t_idx = triples_idx[:, 2]

        h_embed = ent_embed[h_idx] # (B, D)
        r_embed = rel_embed[r_idx]  # (B, D)
        t_embed = ent_embed[t_idx]  # (B, D)

        return torch.stack([h_embed, r_embed, t_embed], dim=1)  # (B, 3, D)

    

    def encode_gnn_subgraphs(self, head_ids, head_id2graph, ent_embed, rel_embed):
        # device = next(self.parameters()).device
        subgraph_list = []
        head_nids = []
        device = ent_embed.device
        for head_id in head_ids:
            triples = head_id2graph[int(head_id)]  # [(h, r, t), ...]

            # 1. Triple → DGLGraph
            graph, head_nid = self.build_dgl_graph_from_triples(triples, head_id)
            subgraph_list.append(graph)
            head_nids.append(head_nid)

        # 2. 마스크 설정
        for g, h_nid in zip(subgraph_list, head_nids):
            g.ndata['head'] = torch.zeros(g.num_nodes(), dtype=torch.bool)
            g.ndata['head'][h_nid] = True
        # 각 subgraph에 초기 임베딩 추가
        # 각 subgraph에 초기 임베딩 추가

        for i, g in enumerate(subgraph_list):
            g = g.to(device)
            g.ndata['id'] = g.ndata['id'].to(device)
            g.ndata['feat'] = ent_embed[g.ndata['id']]
            deg = g.in_degrees().float().clamp(min=1)
            g.ndata['norm'] = torch.pow(deg, -0.5).unsqueeze(1).to(g.device)
            # 'etype'이 없으면 추가해주기
            if 'etype' not in g.edata:
                g.edata['etype'] = g.edata['rel_type'].to(device)

            subgraph_list[i] = g

        batched_graph = dgl.batch(subgraph_list).to(device)

        batched_graph.ndata['h'] = ent_embed[batched_graph.ndata['id']]

        all_node_repr = self.gnn_encoder(batched_graph, ent_embed, rel_embed)
        head_repr = all_node_repr[batched_graph.ndata['head']]  # (B, D)

        return head_repr

    def build_dgl_graph_from_triples(self, triples, head_id):
        """
        triples: List of (h, r, t) in int
        head_id: int
        Returns: (DGLGraph, head_node_index)
        """
        nodes = set()
        for h, _, t in triples:
            nodes.add(h)
            nodes.add(t)

        id_map = {nid: i for i, nid in enumerate(sorted(nodes))}

        src = [id_map[h] for h, _, _ in triples]
        dst = [id_map[t] for _, _, t in triples]
        etype = [r for _, r, _ in triples]

        g = dgl.graph((src, dst), num_nodes=len(id_map))
        g.edata['rel_type'] = torch.tensor(etype, dtype=torch.long)

        # node 'id' feature (original id → for embedding)
        id_tensor = torch.tensor([nid for nid in sorted(nodes)], dtype=torch.long)
        g.ndata['id'] = id_tensor

        head_nid = id_map[int(head_id)]
        return g, head_nid



    def forward(self, text_embed, triples_idx, head_subg_txt_repr, head_id2graph, ent_embed, rel_embed):
        """
        triples_str: list of (h, r, t) in string
        triples_idx: list of (h_id, r_id, t_id) in integer
        """

        # device = next(self.parameters()).device
        # ------------------ Textual Representation -------------------------

        text_embed = torch.cat([text_embed, head_subg_txt_repr.unsqueeze(1)], dim=1) # (B, 4, D)

        # ------------------- Graph Representation -----------------------
        triple_embed = self.encode_kge_triple(triples_idx, ent_embed, rel_embed)
        
        head_ids = triples_idx[:, 0]
        gnn_head_repr = self.encode_gnn_subgraphs(head_ids, head_id2graph, ent_embed, rel_embed).unsqueeze(1) #(B, 1, D)
        triple_embed = torch.cat([triple_embed, gnn_head_repr], dim=1)          # (B, 4, D)
        
        # === Alignment ===
        aligned = self.aligner(text_embed, triple_embed)
        return aligned  # (B, 4, 21504)   
    

class AlignModule_Lite_wo_SubG(nn.Module):
    def __init__(self, ent2id, rel2id, num_heads=8, kge_dim=4096):
        super().__init__()


        # self.triple_text_encoder = TextEncoderWithTripleExtract(self.tokenizer, text_encoder)


        self.aligner = TripleTextAligner(dim=kge_dim, num_heads=num_heads)


    def encode_kge_triple(self, triples_idx, ent_embed, rel_embed):  # triples_idx: Tensor of shape (B, 3)
        """
        triples_idx: torch.LongTensor of shape (B, 3) — (h_id, r_id, t_id)
        returns: (B, 3, D)
        """
        # device = next(self.parameters()).device

        h_idx = triples_idx[:, 0]
        r_idx = triples_idx[:, 1]
        t_idx = triples_idx[:, 2]

        h_embed = ent_embed[h_idx] # (B, D)
        r_embed = rel_embed[r_idx]  # (B, D)
        t_embed = ent_embed[t_idx]  # (B, D)

        return torch.stack([h_embed, r_embed, t_embed], dim=1)  # (B, 3, D)


    def forward(self, text_embed, triples_idx, ent_embed, rel_embed):
        """
        triples_str: list of (h, r, t) in string
        triples_idx: list of (h_id, r_id, t_id) in integer
        """

        # device = next(self.parameters()).device

        # ------------------- Graph Representation -----------------------
        triple_embed = self.encode_kge_triple(triples_idx, ent_embed, rel_embed)
        

        # === Alignment ===
        aligned = self.aligner(text_embed, triple_embed)
        return aligned  # (B, 4, D)   



if __name__ == '__main__' : 
    # === triple 및 verbalize ===
    # triples = [
    #     ("kim", "has a job of", "pilot"),
    #     ("alice", "lives in", "paris"),
    #     ("bob", "is friend of", "john"),
    #     ("new york", "is capital of", "usa"),
    #     ("elon musk", "founded", "spacex"),
    #     ("she", "works at", "google"),
    #     ("john", "is a", "scientist"),
    #     ("maria", "likes", "chocolate cake"),
    #     ("the cat", "sits on", "the mat"),
    #     ("this company", "was established in", "1999"),
    # ]

    # texts = [f"{triple[0]} {triple[1]} {triple[2]}" for triple in triples] # → "kim has a job of pilot"
    

    # === tokenizer, encoder 준비 ===
    # model_name = "EleutherAI/gpt-j-6B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = GPTJForCausalLM.from_pretrained(model_name)

    # encoder = TextEncoderWithTripleExtract(tokenizer, model)
    # selected_embeds, input_ids, position_ids = encoder.encode_triples(triples)
    
    # print("\n📌 Verifying token positions:")
    # for i, (ids, pos) in enumerate(zip(input_ids, position_ids)):
    #     tokens = tokenizer.convert_ids_to_tokens(ids)
    #     h_idx, r_idx, t_idx = pos.tolist()

    #     print(f"\nExample {i+1}: {' '.join(tokens)}")
    #     print(f"  Head     @ {h_idx}: {tokens[h_idx] if h_idx < len(tokens) else 'OUT-OF-BOUND'}")
    #     print(f"  Relation @ {r_idx}: {tokens[r_idx] if r_idx < len(tokens) else 'OUT-OF-BOUND'}")
    #     print(f"  Tail     @ {t_idx}: {tokens[t_idx] if t_idx < len(tokens) else 'OUT-OF-BOUND'}")
    # print(selected_embeds.size())

    datapath = '../dataprocessing/peacok_person_t1_name'
    with open(f"{datapath}/train.txt", 'r') as f :
        train = f.readlines()

    with open("ent2id.pkl", 'rb') as f :
        ent2id = pickle.load(f)
    id2ent = {pair[1] : pair[0] for pair in ent2id.items()}
    with open("rel2id.pkl", 'rb') as f :
        rel2id = pickle.load(f)
    train_ind = []
    for line in train :
        h, r, t = line.strip().split('\t')
        train_ind.append((ent2id[h], rel2id[r], ent2id[t]))
    # =======================================================================
    #count params
    aligner = AlignModule_Lite_wo_SubG(ent2id, rel2id).to('cuda')
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 예시

    print(f"Trainable parameters: {count_parameters(aligner)}")

    # =======================================================================
    with open(f"{datapath}/person_to_name.json", 'r') as f:
        p2n = json.load(f)
    with open("ent2sub_ind_t1.pkl", 'rb') as f :
        head_id2graph = pickle.load(f)

    with open("subgraph_txt_repr_t1.pkl", 'rb') as f :
        subg_txt_repr = pickle.load(f)
    ent_emb = torch.load('entity_embedding_t1.pt').to('cuda')
    rel_emb = torch.load('relation_embedding_t1.pt').to('cuda')

    triple_embeddings = torch.load('triples_txt_embed_t1.pt').to('cuda')

    name2ind = {n : ent2id[n] for n in list(p2n.values())}
    subg_txt_repr_ind = {name2ind[e] : subg_txt_repr[e] for e in subg_txt_repr}

    # align_module = AlignModule(tokenizer, model, 4096, num_heads=8)

    hasajobof = rel2id['has a job of']
    person_triples_txt = []
    person_triples_idx = []
    target_index_of_triple = []
    for i, triple in enumerate(train) :
        h, r, t = triple.strip().split('\t')
        if r == 'has a job of':
            # newline = f"{h}\t{r}\t{t}\n"
            newline = (h, r, t)
            person_triples_txt.append(newline)
            person_triples_idx.append(train_ind[i])
            target_index_of_triple.append(i)



    ind = 245
    triples_str = person_triples_txt[ind : ind+1]
    triples_idx = person_triples_idx[ind : ind+1]
    target_index_of_triple = target_index_of_triple[ind : ind+1]

    print('(1) target triple txt: ', triples_str)
    print('(2) target triple idx: ', triples_idx)
    
    

    head_indices = [triple[0] for triple in triples_idx]
    print('(3) head indices : ', head_indices)

    triples_idx = torch.LongTensor(triples_idx).to('cuda') #(B=1, 3)


    print('(4) ent, rel emb size : ', ent_emb.size(), rel_emb.size())
    
    subg_txt_repr_pt = torch.stack([subg_txt_repr_ind[h] for h in head_indices]).to('cuda') #(B=1, 4096)

    target_index_of_triple = torch.LongTensor(target_index_of_triple).to('cuda')
    
    print('(5) subg txt repr pt : ', subg_txt_repr_pt.size())

    print('(6) target index of triple : ', target_index_of_triple.size())

    
    print(target_index_of_triple)

    text_embedding = triple_embeddings[target_index_of_triple]

    
    # ================ Model Load ===========================
    model_name = "EleutherAI/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    aligner = AlignModule_Lite(ent2id, rel2id).to('cuda')


    
    # graph, init_ent_emb, init_rel_emb = load_graph(train, ent2id, rel2id, gnn_config[1])

    res = aligner(text_embedding, triples_idx, subg_txt_repr_pt, head_id2graph, ent_emb, rel_emb)
    print(res)
    print(res.size())
    
    













