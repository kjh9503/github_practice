import argparse
import fnmatch
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
sys.path.append('lm-evaluation-harness/')
import lm_eval.tasks as tasks
import pickle
import re
import json
import random
import torch
import torch.nn.functional as F
from data import PromptDataset, replace_person_ids

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()

args = parse_args()

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16).cuda()

model.eval()

# Load data 

with open(f'{args.data_name}/train.txt', 'r') as f :
    train = f.readlines() 
prompt_to_full = {}
prompt_to_ans = {}
with open(f'{args.data_name}/person_to_name.json', 'r') as f :
    p2n = json.load(f)
names = list(p2n.values())
prompt_name_order = []
for line in train :
    h, r, t = line.strip().split('\t')
    if h not in names : continue
    if r != 'has a job of' : continue
    prompt = f"{h} has a job of"
    full = f"{h} has a job of {t}"
    ans = t
    prompt_to_full[prompt] = full
    prompt_to_ans[prompt] = ans
    prompt_name_order.append(h)

        

prompt_to_full_file = 'real_train_data_w_prompt_t1_full.json'
prompt_to_ans_file = 'real_train_data_prompt2ans_t1_full.json'
with open(f'{args.data_name}/person_to_name.json', 'r') as f :
    p2n = json.load(f) 
    
train_datalist = []
for i, prompt in enumerate(prompt_to_ans):
    ans = prompt_to_ans[prompt]
    input_ = prompt
    init_text = prompt_to_full[prompt]
    init_label = ans
    person = prompt_name_order[i]
    sample = {'input' : input_, 'init_text' : init_text, 'init_label' : init_label, 'person' : person}
    train_datalist.append(sample)
        

print(len(train_datalist))
dataset = PromptDataset(train_datalist, tokenizer)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

losses = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)
        # print(input_ids[0])
        # print(labels[0])
        # print(batch['sample']['init_text'][0])
        # print()
        # print('-----------------------------------------------------------')
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        losses.append(loss.item())
        # print(loss.item())

        

# Save results
torch.save({'losses': losses, 'samples': train_datalist}, args.out_dir)
