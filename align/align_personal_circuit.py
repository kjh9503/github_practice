import pickle
import json
import math
from copy import deepcopy
import time
import torch
import torch.nn.functional as F
from torch import optim
from transformers import AutoTokenizer
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import AutoModelForCausalLM
from datetime import datetime
from align import AlignModule_Lite_Circuit
from collections import defaultdict
from typing import List
from util import is_correct, normalize_text
import argparse
import os

def load_circuits(circuit_path, num_head, n_circuit_layer=28):
    with open("./circuits/person2id.json", 'r') as f :
        p2i = json.load(f)
    i2p = {pair[1] : pair[0] for pair in p2i.items()}
    person_circuit = {}
    orig_circuits = torch.load(circuit_path)

    for i in trange(len(i2p), desc='loading personal circuits'):
        person = i2p[i]
        person_circuit[person] = {}
        info = orig_circuits[person]['path']
        for layer_idx, heads in info :
            if len(person_circuit[person]) == n_circuit_layer :
                break
            person_circuit[person][layer_idx] = heads[:num_head]

    return person_circuit
# ---------- 설정 ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Injection Layer Configuration")

    parser.add_argument('--inj_layer', type=int, required=True,
                        help='The index of the layer to inject into.')
    parser.add_argument('--num_edit', type=int, required=True,
                        help='The number of edits to apply.')
    parser.add_argument('--epoch', type=int, required=True,
                        help='Number of training epochs.')
    parser.add_argument('--model_name', type=str, default='../finetuned_models/gpt-j-6B_t1_chain_name_gptj_full')
    
    parser.add_argument('--locality_test', action='store_true', default=False)

    parser.add_argument('--kl_factor', required=True, type=float)

    parser.add_argument('--decay_factor', required=True, type=float)



    parser.add_argument('--note', default='', type=str)

    parser.add_argument('--num_circuit_head', required=True, type=int)

    parser.add_argument('--num_circuit_layer', required=True, type=int)

    parser.add_argument('--circuit_path', type=str, required=True)

    parser.add_argument('--lr', default=1e-5, type=float)


    parser.add_argument('--time', required=True, type=str)

    return parser.parse_args()

args = parse_args()

# ---------- 설정 ----------
print("======= Argument Summary =======")
for key, value in vars(args).items():
    print(f"{key}: {value}")


MODEL_NAME = args.model_name

DEVICE = "cuda"
LR = args.lr
ATTN_LAYER_INDEX = args.inj_layer


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side='left'

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)

hidden_size = model.config.hidden_size #4096
n_heads = model.config.n_head #16
head_dim = hidden_size // n_heads #256


model.eval()


# -------------데이터 로드 ------------
datapath1 = '../data/peacok_person_t1_name'
datapath2 = f'../data/peacok_person_{args.time}_name'


with open(f"{datapath1}/train.txt", 'r') as f :
    train_1 = f.readlines()
with open(f"{datapath2}/train.txt", 'r') as f :
    train_2 = f.readlines()
with open("ent2id.pkl", 'rb') as f :
    ent2id = pickle.load(f)


id2ent = {pair[1] : pair[0] for pair in ent2id.items()}

with open("rel2id.pkl", 'rb') as f :
    rel2id = pickle.load(f)

train_ind_1 = []
for line in train_1 :
    h, r, t = line.strip().split('\t')
    h_i = ent2id[h]
    r_i = rel2id[r]
    t_i = ent2id[t]
    train_ind_1.append((h_i, r_i, t_i))

train_ind_2 = []
for line in train_2 :
    h, r, t = line.strip().split('\t')
    h_i = ent2id[h]
    r_i = rel2id[r]
    t_i = ent2id[t]
    train_ind_2.append((h_i, r_i, t_i))


with open(f"{datapath1}/person_to_name.json", 'r') as f:
    p2n = json.load(f)
n2p = {pair[1] : pair[0] for pair in p2n.items()}
with open("./data/ent2sub_ind_t1.pkl", 'rb') as f :
    head_id2graph_t1 = pickle.load(f)
with open(f"./data/ent2sub_ind_{args.time}.pkl", 'rb') as f :
    head_id2graph_t2 = pickle.load(f)


triple_embeddings_1 = torch.load('./data/triples_txt_embed_t1_llm_t1_chain_gpt.pt').to(DEVICE)
triple_embeddings_2 = torch.load(f'./data/triples_txt_embed_{args.time}_llm_t1_chain_gpt.pt').to(DEVICE)

with open("./data/subgraph_txt_repr_t1_llm_t1_chain_gpt.pkl", 'rb') as f :
    subg_txt_repr_1 = pickle.load(f)
with open(f"./data/subgraph_txt_repr_{args.time}_llm_t1_chain_gpt.pkl", 'rb') as f :
    subg_txt_repr_2 = pickle.load(f)


ent_emb_1 = torch.load('./data/entity_embedding_t1.pt').to(DEVICE)
rel_emb_1 = torch.load('./data/relation_embedding_t1.pt').to(DEVICE)

ent_emb_2 = torch.load(f'./data/entity_embedding_{args.time}.pt').to(DEVICE)
rel_emb_2 = torch.load(f'./data/relation_embedding_{args.time}.pt').to(DEVICE)


D = ent_emb_1.size(1)

name2ind = {n : ent2id[n] for n in list(p2n.values())}

names = list(name2ind.keys())


subg_txt_repr_ind_1 = {name2ind[e] : subg_txt_repr_1[e] for e in subg_txt_repr_1}
subg_txt_repr_ind_2 = {name2ind[e] : subg_txt_repr_2[e] for e in subg_txt_repr_2}


if args.locality_test :
    with open(f'./data/loc_{args.time}/locality_{args.time}.json', 'r') as f :
        loc = json.load(f)
    with open(f'./data/loc_{args.time}/locality_triple_txts_{args.time}.json', 'r') as f :
        locality_triple_txts = json.load(f)
    with open(f'./data/loc_{args.time}/locality_triple_inds_{args.time}.json', 'r') as f :
        locality_triple_inds = json.load(f)

    locality_data = list((d['prompt'], d['answer']) for d in loc)

    with open(f'./data/loc_{args.time}/ent2sub_ind_loc_t1.pkl', 'rb') as f :
        head_id2graph_t1_loc = pickle.load(f)
    with open(f'./data/loc_{args.time}/ent2sub_ind_loc_{args.time}.pkl', 'rb') as f :
        head_id2graph_t2_loc = pickle.load(f)

    
    with open(f'./data/loc_{args.time}/subgraph_txt_repr_t1_llm_t1_chain_gpt.pkl', 'rb') as f :
        subgraph_txt_repr_1_loc = pickle.load(f)
    with open(f'./data/loc_{args.time}/subgraph_txt_repr_{args.time}_llm_t1_chain_gpt.pkl', 'rb') as f :
        subgraph_txt_repr_2_loc = pickle.load(f)
    
    subg_txt_repr_ind_loc_1 = {ent2id[e] : subgraph_txt_repr_1_loc[e] for e in subgraph_txt_repr_1_loc}

    subg_txt_repr_ind_loc_2 = {ent2id[e] : subgraph_txt_repr_2_loc[e] for e in subgraph_txt_repr_2_loc}

    triple_embeddings_loc = torch.load(f'./data/loc_{args.time}/triples_txt_embed_loc_{args.time}_llm_t1gpt_chain.pt')

# -------------데이터 로드 ------------


hr2t_idx_t2 = defaultdict(list)
for i, triple in enumerate(train_2):
    h, r, t = triple.strip().split('\t')
    if h in names:
        hr2t_idx_t2[(h, r)].append((t, i)) 

hasajobof = rel2id['has a job of']

person_triples_txt_1 = []
person_triples_txt_2 = []
person_triples_idx_1 = []
person_triples_idx_2 = []

target_id_1 = []
target_id_2 = []
for i, triple in enumerate(train_1):
    h, r, t = triple.strip().split('\t')
    if h in names and (h, r) in hr2t_idx_t2:

        t2, i2 = hr2t_idx_t2[(h, r)].pop(0)
        if t == t2 :
            continue

        person_triples_txt_1.append((h, r, t))
        person_triples_txt_2.append((h, r, t2))

        person_triples_idx_1.append([ent2id[h], rel2id[r], ent2id[t]])
        person_triples_idx_2.append([ent2id[h], rel2id[r], ent2id[t2]])

        target_id_1.append(i)
        target_id_2.append(i2)

        # 더 이상 없으면 제거
        if not hr2t_idx_t2[(h, r)]:
            del hr2t_idx_t2[(h, r)]

assert len(person_triples_txt_1) == len(person_triples_txt_2) == len(person_triples_idx_1) == \
len(person_triples_idx_2)  == len(target_id_1) == len(target_id_2),\
(len(person_triples_txt_1), len(person_triples_txt_2) , len(person_triples_idx_1), \
len(person_triples_idx_2) , len(target_id_1) ,len(target_id_2))


for i, (trip1, trip2) in enumerate(zip(person_triples_txt_1, person_triples_txt_2)):
    h1, r1, t1 = trip1
    h2, r2, t2 = trip2
    assert h1 == h2 and r1 == r2, f"[Index {i}] Mismatch in head or relation: {h1, r1} vs {h2, r2}"

    if t1 != t2:
        print(f"[Index {i}] Matched HR: ({h1}, {r1}) | t1: {t1} ≠ t2: {t2}")


personal_circuits = load_circuits(args.circuit_path, args.num_circuit_head, n_circuit_layer=args.num_circuit_layer)
print(len(personal_circuits))
align_module = AlignModule_Lite_Circuit(ent2id, rel2id, num_circuit_head = args.num_circuit_head, num_layer=args.num_circuit_layer)  # 실제 정의한 constructor에 맞춰 넣기



align_module.to(DEVICE)
optimizer = optim.Adam(align_module.parameters(), lr=LR)


# ---------- Forward ----------
align_module.train()

num_edit=args.num_edit
total_data = len(person_triples_txt_1)
total_batches = math.ceil(total_data / num_edit)
epoch = args.epoch


correct = 0

if args.locality_test :
    if 'chain' in args.model_name :
        dir_name = f'log/batch_chain_PerCir(L{args.num_circuit_layer}H{args.num_circuit_head})_{args.time}_essence_{args.kl_factor}_decay_{args.decay_factor}_locality_L{ATTN_LAYER_INDEX}_N{num_edit}_E{epoch}_{args.note}'
    else :
        dir_name = f'log/batch_PerCir(L{args.num_circuit_layer}H{args.num_circuit_head})_{args.time}_essence_{args.kl_factor}_decay_{args.decay_factor}_locality_L{ATTN_LAYER_INDEX}_N{num_edit}_E{epoch}_{args.note}'
else :
    raise NotImplementedError(f'Not implemented for args.locality test : {args.locality_test}')

if not os.path.exists(dir_name):
    os.makedirs(dir_name)
if args.locality_test :
    loc_dir = f"{dir_name}/locality_log"
    if not os.path.exists(loc_dir) :
        os.makedirs(loc_dir)

with open(f'{dir_name}/config.json', 'w') as f:
    json.dump(vars(args), f, indent=4)

train_fname = f'{dir_name}/batch_train_log.txt'
with open(train_fname, 'w', encoding='utf-8') as f :
        f.write('==========================TRAIN LOG==================================\n')


for batch_num in range(total_batches):
    fname = f'{dir_name}/generation_log_batch_{batch_num}.txt'
    # if batch_num != 144:
    #     continue
    with open(fname, 'w') as f :
        f.write('==========================GENERATION LOG==================================\n')

    
    align_module = AlignModule_Lite_Circuit(ent2id, rel2id, num_circuit_head = args.num_circuit_head, num_layer=args.num_circuit_layer).to(DEVICE)  # 실제 정의한 constructor에 맞춰 넣기

    # align_module = torch.nn.DataParallel(AlignModule_Lite_Circuit(ent2id, rel2id))
    # align_module.to('cuda')

    optimizer = optim.Adam(align_module.parameters(), lr=LR)
    
    start_idx = batch_num * num_edit
    end_idx = min((batch_num + 1) * num_edit, total_data)

    batch_indices = list(range(start_idx, end_idx))
 
    # total_loss, loss_nll, kl_loss, weight_decay = torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])
    # hook_handle = None
    best_loss = 1000
    early_stop_patient = 0
    for e in range(epoch):
    
        triples_idx_1 = torch.tensor([person_triples_idx_1[i] for i in batch_indices]).to(DEVICE)
        triples_idx_2 = torch.tensor([person_triples_idx_2[i] for i in batch_indices]).to(DEVICE) #(16, 3)
        head_ids = triples_idx_2[:, 0].tolist()

        subg_txt_repr_pt_1 = torch.stack([subg_txt_repr_ind_1[hid] for hid in head_ids]).to(DEVICE) #(16, 4096)
        subg_txt_repr_pt_2 = torch.stack([subg_txt_repr_ind_2[hid] for hid in head_ids]).to(DEVICE)

        text_embedding_1 = torch.stack([triple_embeddings_1[i] for i in batch_indices]).to(DEVICE) #(16, 3, 4096)
        text_embedding_2 = torch.stack([triple_embeddings_2[i] for i in batch_indices]).to(DEVICE)


        texts_2 = [f"{person_triples_txt_2[i][0]} {person_triples_txt_2[i][1]} {person_triples_txt_2[i][2]}" for i in batch_indices]

        input_ids_2 = tokenizer(texts_2, return_tensors="pt", max_length=25, padding='max_length', truncation=True).to(DEVICE)

        labels = input_ids_2.input_ids.clone()

        for j, i in enumerate(batch_indices):
            tail_tok = tokenizer(' ' + person_triples_txt_2[i][2], add_special_tokens=False)['input_ids']
            labels[j, :-len(tail_tok)] = -100

        attention_mask = input_ids_2.attention_mask

        with torch.no_grad():
            logits_pre = model(input_ids=input_ids_2.input_ids, attention_mask=attention_mask).logits
            kl_logits_pre = logits_pre[:, -len(tail_tok):, :]  
            kl_log_probs_pre = torch.nn.functional.log_softmax(kl_logits_pre, dim=-1).detach()
        
        repr_t1 = align_module(text_embedding_1, triples_idx_1, subg_txt_repr_pt_1, head_id2graph_t1, ent_emb_1, rel_emb_1)
        repr_t2 = align_module(text_embedding_2, triples_idx_2, subg_txt_repr_pt_2, head_id2graph_t2, ent_emb_2, rel_emb_2) #(16, 4, 21504)
        
        delta_vec =  torch.nn.functional.sigmoid(repr_t2[:, 0, :] - repr_t1[:, 0, :]).unsqueeze(1) #(bs, 21504) : 256 * 28 * 3
    
        def make_hook(layer_idx : int, head_mask : List[int], activation): # (bs, 1, 21504)
            assert len(head_mask) == args.num_circuit_head
            def attn_hook(module, input, output) :
                attn_out, attn_weights = output
                for i, head in enumerate(head_mask) :
                    global_head_index = layer_idx * len(head_mask) + i
                    activation_slice = activation[:, :, global_head_index * head_dim : (global_head_index + 1) * head_dim] #(b, 1, 256)
                    
                    attn_out[:, :, head * head_dim : (head + 1) * head_dim] += activation_slice / (2+torch.exp(torch.tensor([layer_idx], device=activation_slice.device)))
                return (attn_out, attn_weights)
            return attn_hook

        hook_handlers = []
        
        # ==========================================================================================
        circuits = personal_circuits[person_triples_txt_2[i][0]]
        # ==========================================================================================
        for layer_idx in list(circuits.keys()) :
            head_idxs = circuits[layer_idx]
            layer_module = model.transformer.h[layer_idx].attn
            hook = layer_module.register_forward_hook(make_hook(layer_idx, head_idxs, delta_vec))
            hook_handlers.append(hook)
        

        # 5. training step
        align_module.train()
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids_2.input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        
        logits_post = outputs.logits
        loss_nll = outputs.loss
        
        kl_logits_post = logits_post[:, -len(tail_tok):, :]
        kl_log_probs_post = torch.nn.functional.log_softmax(kl_logits_post, dim=-1)

        kl_loss = torch.nn.functional.kl_div(
            kl_log_probs_post, kl_log_probs_pre, log_target=True, reduction='batchmean'
        )
        # ============== weight decay ============================
        delta_norm = torch.norm(delta_vec, p=2)
        repr_norm = torch.norm(repr_t1[:, 0, :], p=2)
        weight_decay = (delta_norm / (repr_norm + 1e-8)) ** 2
        # ============== weight decay ============================
        total_loss = loss_nll + args.kl_factor * kl_loss + args.decay_factor * weight_decay

        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [Batch {batch_num}] | Epoch {e}] Loss: {total_loss.item():.4f} = {loss_nll.item():.4f} + {(args.kl_factor * kl_loss).item():.4f} + {(args.decay_factor * weight_decay).item():.4f}")

        with open(train_fname, 'a', encoding='utf-8') as f :
            f.write(f"[{now}] [Batch {batch_num}] | Epoch {e}] Loss: {total_loss.item():.4f} = {loss_nll.item():.4f} + {(args.kl_factor * kl_loss).item():.4f} + {(args.decay_factor * weight_decay).item():.4f}\n")
        
        total_loss.backward()

        optimizer.step()
        if loss_nll.item() <= best_loss:
            best_loss = loss_nll.item()
            early_stop_patient = 0
            best_model = model.state_dict()
            best_align_module = align_module.state_dict()
        elif loss_nll.item() < 0.05 :
            early_stop_patient += 1
        
        if loss_nll.item() <= 0.1 :
            break
        if e < epoch-1:
            for hook_handle in hook_handlers :
                hook_handle.remove()

        
    with open(train_fname, 'a', encoding='utf-8') as f :
        f.write(f"---------------------------------------------------------------------------\n")
        
    ##############################
    ####### 6. generation  #######
    ##############################
    hr_texts = [f"{person_triples_txt_2[i][0]} {person_triples_txt_2[i][1]}" for i in batch_indices]
    input_ids_hr = tokenizer(hr_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    input_ids_full = input_ids_hr['input_ids']
    attention_mask_full = input_ids_hr['attention_mask']


    hr_output = model.generate(
        input_ids=input_ids_full,
        attention_mask=attention_mask_full,
        max_new_tokens=10,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


    decoded_texts = tokenizer.batch_decode(hr_output, skip_special_tokens=True)

    prompt_lens = attention_mask_full.sum(dim=1)  # shape: (batch_size,)

    tails_1 = [person_triples_txt_1[i][2] for i in batch_indices]
    tails_2 = [person_triples_txt_2[i][2] for i in batch_indices]
    
    batch_corr = 0
    for i, (full_text, t2) in enumerate(zip(decoded_texts, tails_2)):
        cor = is_correct(full_text, t2)
        correct += cor
        batch_corr += cor
        now = datetime.now().strftime("%H:%M:%S")
        # print(f"[{now}] [Batch {batch_num} | Last Epoch Loss: {total_loss.item():.4f} | Predicted: {full_text.strip()} | GT_T1: {tails_1[i]} | GT_T2: {tails_2[i]}")
        print(f"[{now}] [Batch {batch_num}] | Last Epoch Loss: {total_loss.item():.4f} = {loss_nll.item():.4f} + {(args.kl_factor * kl_loss).item():.4f} + {(args.decay_factor * weight_decay).item():.4f} | Predicted: {full_text.strip()} | GT_T1: {tails_1[i]} | GT_T2: {tails_2[i]}")
        print(f'-------------------> {cor}')
        print('--------------------------------------------------------------------------------')
        with open(fname, 'a', encoding='utf-8') as f :
            f.write(f"[{now}] [Batch {batch_num}] | Last Epoch Loss: {total_loss.item():.4f} = {loss_nll.item():.4f} (nll) + {(args.kl_factor * kl_loss).item():.4f} (kl) + {(args.decay_factor * weight_decay).item():.4f} (wd) | Predicted: {full_text.strip()} | GT_T1: {tails_1[i]} | GT_T2: {tails_2[i]}\n")
            f.write(f'-------------------> {cor}\n')
            f.write('--------------------------------------------------------------------------------------\n')
    print(f"[{now}] [Batch {batch_num}] number of correct : {batch_corr}")
    with open(fname, 'a', encoding='utf-8') as f :
        f.write(f"[{now}] [Batch {batch_num}] number of correct : {batch_corr}\n")
    
    # if hook_handle :
    for hook_handle in hook_handlers :
        hook_handle.remove()

    if args.locality_test :
        def batch_generate_all(batch_size=32, max_new_tokens=20):
            all_generated = []

            total_data = len(loc)
            total_batches = (total_data + batch_size - 1) // batch_size

            for b in trange(total_batches, desc='Locality Test'):
                batch_start = b * batch_size
                batch_end = min((b + 1) * batch_size, total_data)

                batch_prompts = [d['prompt'] for d in loc[batch_start:batch_end]]
                batch_idx = torch.LongTensor(locality_triple_inds[batch_start:batch_end]).to(DEVICE)


                text_embedding = triple_embeddings_loc[batch_start:batch_end].to(DEVICE)


                head_ids = batch_idx[:, 0].tolist()
                subg_txt_repr_pt_1_loc = torch.stack([subg_txt_repr_ind_loc_1[hid] for hid in head_ids]).to(DEVICE)
                subg_txt_repr_pt_2_loc = torch.stack([subg_txt_repr_ind_loc_2[hid] for hid in head_ids]).to(DEVICE)


                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=25)
                input_ids = inputs['input_ids'].to(DEVICE)
                attn_mask = inputs['attention_mask'].to(DEVICE)

                with torch.no_grad():
                    repr_t1 = align_module(text_embedding, batch_idx, subg_txt_repr_pt_1_loc,
                                        head_id2graph_t1_loc, ent_emb_1, rel_emb_1)  # (B, 4, D)
                    repr_t2 = align_module(text_embedding, batch_idx, subg_txt_repr_pt_2_loc,
                                        head_id2graph_t2_loc, ent_emb_2, rel_emb_2)  # (B, 4, D)
                    delta_cache = torch.nn.functional.sigmoid(repr_t1[:, 0, :] - repr_t2[:, 0, :]).unsqueeze(1)  # (B, 1, D)


                def make_hook(layer_idx : int, head_mask : List[int], activation):
                    assert len(head_mask) == args.num_circuit_head
                    def attn_hook(module, input, output) :
                        attn_out, attn_weights = output
                        for i, head in enumerate(head_mask) :
                            global_head_index = layer_idx * len(head_mask) + i
                            activation_slice = activation[:, :, global_head_index * head_dim : (global_head_index + 1) * head_dim]
                            # activation_slice = activation_slice #(b, 1, 256)
                            attn_out[:, :, head * head_dim : (head + 1) * head_dim] += activation_slice / (2+torch.exp(torch.tensor([layer_idx], device=activation_slice.device)))
                        return (attn_out, attn_weights)
                    return attn_hook
                
                hook_handlers = []
                for layer_idx, head_idxs in circuits.items() :
                    layer_module = model.transformer.h[layer_idx].attn
                    hook = layer_module.register_forward_hook(make_hook(layer_idx, head_idxs, delta_cache))
                    hook_handlers.append(hook)
            

                with torch.no_grad():


                    outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=20,
                    min_new_tokens=5,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    )


                for hook_handle in hook_handlers :
                    hook_handle.remove()

                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_generated.extend(generated_texts)



            return all_generated
        
        results = batch_generate_all(batch_size=64)
        
        loc_correct = 0
        loc_total = len(loc)

        with open(f'{loc_dir}/locality_generation_batch_num_{batch_num}.txt', 'w') as f :
            for triple, generated in zip(loc, results):
                target = triple['answer']
                print(f"[{triple['prompt']} | {target}] → {generated}")
                
                f.write(f"[{triple['prompt']} | {target}] → {generated}\n")
                if is_correct(generated, target):
                    loc_correct += 1
                    print('--------> True')
                    f.write('--------> True\n')
                else :
                    print('--------> False')
                    f.write('--------> False\n')
                print('-----------------------------------------------------------')
                f.write('-----------------------------------------------------------\n')


        loc_accuracy = loc_correct / loc_total * 100
        print(f"Locality Accuracy: {loc_accuracy:.2f}% ({loc_correct}/{loc_total})")
        with open(f'{loc_dir}/locality_generation_batch_num_{batch_num}.txt', 'a') as f :
            f.write(f"Locality Accuracy: {loc_accuracy:.2f}% ({loc_correct}/{loc_total})\n")



print(f'{correct}/{total_data}')
with open(fname, 'a') as f :
    f.write(f'Last epoch acc : {correct}/{total_data}\n')