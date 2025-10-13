import re
import json
import pickle
import random
import torch
from torch.utils.data import Dataset

class PeacokDataset_NLQ(Dataset):
    def __init__(self, data, tokenizer, prompt_map, max_length=128):
        self.tokenizer = tokenizer
        self.data = list(data.items()) if isinstance(data, dict) else data
        self.max_length = max_length
        self.prompt_map = prompt_map
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, ans = self.data[idx]
        prompt_text = q
        answer_text = ans
        
        full_input = f"{prompt_text} {answer_text}"
        encoding = self.tokenizer(
            full_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        labels = input_ids.clone()
        seq_len = attention_mask.sum().item()
        prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))
        content_start = len(input_ids) - seq_len
        prompt_end = content_start + prompt_len
        labels[:prompt_end] = -100
        labels[attention_mask == 0] = -100   # 패딩 영역도 마스킹
        labels[labels == self.tokenizer.eos_token_id] = -100


        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, 'full_input' : full_input}

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    
    
def load_data(data_path) :
    with open(f'{data_path}/1p_fol_qa_v2.pkl', 'rb') as f: onep = pickle.load(f)
    with open(f'{data_path}/2p_fol_qa_v2.pkl', 'rb') as f: twop = pickle.load(f)
    with open(f'{data_path}/3p_fol_qa_v2.pkl', 'rb') as f: threep = pickle.load(f)
    fols = {**onep, **twop, **threep}
    
    with open(f'{data_path}/data2relpath.pkl', 'rb') as f: data2relpath = pickle.load(f)
    with open(f'{data_path}/relpath2statements.pkl', 'rb') as f: relpath2statement = pickle.load(f)
    
    with open(f'{data_path}/real_train_data_prompt2ans_chain_t1_name.json', 'r') as f : prompt2ans_chain = json.load(f)
    nlq2ans = {prompt : prompt2ans_chain[prompt]['prompt2ans'][prompt] for prompt in prompt2ans_chain}
    with open(f'{data_path}/rel2data_v2.pkl', 'rb') as f : rel2data = pickle.load(f)
    
    data = []
    for qtype in rel2data :
        rel_paths = rel2data[qtype]
        for rel_path in rel_paths :

            fol_queries = rel_paths[rel_path]
            prompt = relpath2statement[rel_path]
            for q in fol_queries :
                first_term = q.split('&')[0].strip()
                entity = re.search(r'\(([^,]+),', first_term).group(1).strip()
                prompt_text = prompt.format(Subject=entity) 

                uniq_ans = nlq2ans[prompt_text]
                uniq_ans = uniq_ans.split('<|endoftext|')[0].strip()
                #t1 finetune 할 때0
                # assert uniq_ans in [ans.replace('.', '') for ans in fols[q]] , (q, uniq_ans, [ans.replace('.', '') for ans in fols[q]])
                #t2 finetune 할 떄
                # =========================================

                chain_info = prompt2ans_chain[prompt_text]
                if chain_info['qtype'] == '1p' :
                    pass
                elif chain_info['qtype'] == '2p' :
                    chain_prompt_1 = chain_info['chain1']
                    interm1 = chain_info['intermediate1']
                    chain_prompt_2 = chain_info['chain2']
                    data.append((chain_prompt_1, interm1))
                    data.append((chain_prompt_2, uniq_ans))
                elif chain_info['qtype'] == '3p' :
                    chain_prompt_1 = chain_info['chain1']
                    interm1 = chain_info['intermediate1']
                    chain_prompt_2 = chain_info['chain2']
                    interm2 = chain_info['intermediate2']
                    chain_prompt_3 = chain_info['chain3']
                    data.append((chain_prompt_1, interm1))
                    data.append((chain_prompt_2, interm2))
                    data.append((chain_prompt_3, uniq_ans))

                data.append((prompt_text, uniq_ans))                
                
                # ====================================

        
    return data, fols, relpath2statement, data2relpath

