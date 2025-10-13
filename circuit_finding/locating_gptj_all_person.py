# GPT-2 compatible version of the original LLaMA head-masking evaluation code (fixed split/merge heads)
import argparse
import fnmatch
import os
import sys
import torch
import time
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from accelerate import Accelerator
from types import MethodType
from tqdm import trange, tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
sys.path.append('lm-evaluation-harness/')
import lm_eval.tasks as tasks

from transformers import DataCollatorForSeq2Seq

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # features는 dict list
        persons = [f.pop("person") for f in features]
        batch = super().__call__(features)
        batch["person"] = persons
        return batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default="./results/")
    parser.add_argument("--sample_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument('--train_on_inputs', default=False, action='store_true')
    parser.add_argument('--add_eos_token', default=False, action='store_true')
    parser.add_argument("--chosen_layers", type=str, default="0-11")
    parser.add_argument("--head_num", type=int, default=4)
    # ====================================================================================
    parser.add_argument('--model_dir', type=str, required=True)
    # ====================================================================================

    return parser.parse_args()


def evaluate(model, batchs):
    losses = []
    accelerator = Accelerator()
    model.eval()
    with torch.no_grad():
        for batch in batchs:
            outputs = model(**batch)
            losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.stack(losses))
  
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()




def masking(model, batchs, layers_w_heads=None):
    hooks = []

    for layer_idx, corrupt_heads in layers_w_heads:
        attn_module = model.transformer.h[layer_idx].attn
        num_heads = attn_module.num_attention_heads
        head_dim = attn_module.embed_dim // num_heads

        def make_hook(corrupt_heads, num_heads, head_dim):
            def hook(module, input, output):
                attn_output, present = output  # ✅ GPT-J는 tuple 반환

                bsz, seq_len, hidden_dim = attn_output.shape
                assert hidden_dim == num_heads * head_dim

                attn_output = attn_output.view(bsz, seq_len, num_heads, head_dim)
                for h in corrupt_heads:
                    attn_output[:, :, h, :] = 0
                attn_output = attn_output.view(bsz, seq_len, hidden_dim)

                return (attn_output, present)  # ✅ 다시 tuple로 묶어서 반환
            return hook

        hook_handle = attn_module.register_forward_hook(make_hook(corrupt_heads, num_heads, head_dim))
        hooks.append(hook_handle)

    loss, perplexity = evaluate(model, batchs)
    for h in hooks:
        h.remove()

    return loss

def layer_conditioned_locating(model, batchs, head_num=4, chosen_layers=[0, 11], person='person'):
    table = []
    path = []
    for layer_idx in range(chosen_layers[0], chosen_layers[1] + 1):
        row = []
        for head_idx in range(model.config.n_head):
            r = masking(model, batchs, layers_w_heads=path + [(layer_idx, [head_idx])])
        
            row.append(r)
        row_tensor = torch.tensor(row)
        table.append(row_tensor)
        topk_heads = torch.topk(row_tensor, k=head_num).indices
        path += [(layer_idx, topk_heads.tolist())]
    return dict(scores=torch.stack(table).T.cpu(), path=path)

def main():

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    

    # =======================================================================================
    def my_generate_and_tokenize_prompt(data_point):
        full_prompt = data_point['init_text']
        prompt = data_point['input']
        
        tokenized_full_prompt = tokenizer(full_prompt, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        
        input_ids = tokenized_full_prompt.input_ids.squeeze(0)
        attention_mask = tokenized_full_prompt.attention_mask.squeeze(0)
        labels = input_ids.clone()

        seq_len = attention_mask.sum().item()
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        content_start = len(input_ids) - seq_len
        prompt_end = content_start + prompt_len

        labels[:prompt_end] = -100
        labels[attention_mask==0] = -100
        labels[labels==tokenizer.eos_token_id] = -100

        tokenized_full_prompt['input_ids'] = input_ids
        tokenized_full_prompt['attention_mask'] = attention_mask
        tokenized_full_prompt['labels'] = labels
        tokenized_full_prompt['person'] = data_point['person']


        return tokenized_full_prompt
    # =======================================================================================

    info = torch.load(args.sample_dir)
    samples = info['samples']
    losses = info['losses']
    persons = [sample['person'] for sample in samples]
    indices = list(range(len(samples)))

    train_datalist = [samples[i] for i in indices]

    train_dataset = Dataset.from_list(train_datalist)
    train_dataset = train_dataset.map(my_generate_and_tokenize_prompt)
    train_dataset = train_dataset.remove_columns(['input', 'init_text', 'init_label'])

    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long).to(device),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long).to(device),
            "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long).to(device),
            "person": [x["person"] for x in batch],  # string이니까 그냥 CPU에 놔둬도 됨
        }

    # Step 3. DataLoader 만들기
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Step 4. batchs 리스트 만들기
    batchs = [batch for batch in dataloader]
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    person2id = {}
    model.eval()
    for i, batch in tqdm(enumerate(batchs), total=len(batchs)) :
        assert len(batch['person']) == 1
        person = batch['person'][0]
        person2id[person] = i
        with torch.no_grad():
            results = layer_conditioned_locating(
                model, [batch],
                head_num=args.head_num,
                chosen_layers=[int(_) for _ in args.chosen_layers.split('-')],
                person=person
            )
        knowledge_circuit = {layer_idx: {'retain_heads': retain_head} for layer_idx, retain_head in results['path']}
        results["knowledge_circuit"] = knowledge_circuit
        torch.save(results, os.path.join(args.save_path, f"info_{i}.pt"))

        
    with open('person2id.json', 'w') as f :
        json.dump(person2id, f)
    print(person2id)

if __name__ == "__main__":
    main()
