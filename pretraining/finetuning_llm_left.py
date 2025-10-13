import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
import random
import pickle
import json
from tqdm import tqdm
import re
from torch import nn
import argparse
import os
from data import PeacokDataset_NLQ, load_data


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text

def is_correct(pred, targets):
    norm_pred = normalize_text(pred)
    for target in targets :
        norm_target = normalize_text(target)

        if norm_target in norm_pred:
            return True

    return False

def evaluate_batch(model, tokenizer, data, printinfo=False, max_length=128, batch_size=64, LOCALITY=False):
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()

    test_data = list(data.items()) if isinstance(data, dict) else data

    correct = 0
    total = 0
    generation = []
    if LOCALITY and (not isinstance(test_data[0], dict)) :
        test_data = [{'prompt' : d[0], 'attribute' : d[1]} for d in test_data]

    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
            batch = test_data[i:i+batch_size]
            prompts, label_sets = [], []

            for q, ans_set in batch:
                rel_path = data2relpath[q]
                prompt = relpath2statement[rel_path]
                entity = re.search(r'\(([^,]+),', q.split('&')[0]).group(1).strip()

                prompts.append(prompt.format(Subject=entity))
                label_sets.append(ans_set)

            

            encoding = tokenizer(prompts, max_length=max_length, truncation=True, padding=True, return_tensors="pt").to(model.device)
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
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

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for prompt, full_text, labels in zip(prompts, decoded, label_sets):
                predicted = full_text[len(prompt):].strip().lower()
                label_texts = {tokenizer.decode(tokenizer.encode(a), skip_special_tokens=True).strip().lower() for a in labels}
                if printinfo:
                    print('---------------------------------------------------------------')
                    print(f"[Prompt] {prompt}\n[Pred] {predicted}\n[Labels] {label_texts}")
                
                generation.append('---------------------------------------------------------------\n')
                generation.append(f"[Prompt] {prompt}\n[Pred] {predicted}\n[Labels] {label_texts}\n")

                if is_correct(predicted, label_texts):
                    correct += 1

                    if printinfo :
                        print('--> True')
                        print('---------------------------------------------------------------')
                    
                    generation.append('--> True\n')
                    generation.append('---------------------------------------------------------------\n')
                
                else :
                    if printinfo :
                        print('--> False')
                        print('---------------------------------------------------------------')
                    generation.append('--> False\n')
                    generation.append('---------------------------------------------------------------\n')
                
                total += 1

    acc = 100 * correct / total if total > 0 else 0
    
    return acc, generation


def train(args):
    if args.save_dir == '' :
        model_name = args.model_name.split('/')[-1]
        save_dir = f'./finetuned_models/{model_name}_{args.note}' #./finetuned_models/gpt-j-6B_t1_chain_name_gptj_full_RE
    else :
        save_dir = args.save_dir
    
    if not os.path.exists(save_dir) :
        os.makedirs(save_dir)
    log_path = os.path.join(save_dir, 'train_log.txt') #./finetuned_models/peacok_nlq_right_gpt2_t2_citcuit/train_log.txt


    tokenizer = AutoTokenizer.from_pretrained(args.load_dir or args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
        
        
    model = AutoModelForCausalLM.from_pretrained(
        args.load_dir or args.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    global relpath2statement, data2relpath
    
    data, fols, relpath2statement, data2relpath = load_data(args.data_path)
    
    print('Number of data : ', len(data))
    
    dataset = PeacokDataset_NLQ(data, tokenizer, relpath2statement)
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PeacokDataset_NLQ.collate_fn)
        
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # if args.accel :
    #     accelerator = Accelerator()
    #     model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    print("Start training...")

    best_acc = 0

    early_stop_patience = args.patience
    best_loss = float('inf')
    no_improve = 0

    with open(log_path, 'w') as log_file : # ./finetuned_models/peacok_nlq_right_gpt2_t2_citcuit/train_log.txt
        for epoch in range(args.start, args.epochs):
                    
            model.train()
            total_loss = 0

            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'

            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PeacokDataset_NLQ.collate_fn)

            for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):

                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["labels"].to(args.device)


                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                # if args.accel :
                #     accelerator.backward(loss)
                # else :
                loss.backward()
                    
                optimizer.step()

                total_loss += loss.item()

            
            avg_loss = total_loss / len(loader)

            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
            
            log_file.write(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}\n")
            log_file.flush()

            if avg_loss < best_loss - 1e-4 :
                best_loss = avg_loss
                no_improve = 0

            else :
                no_improve += 1
            
            if no_improve >= early_stop_patience :
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

            if (epoch + 1) % args.eval_steps == 0:
                
                acc, generation = evaluate_batch(model, tokenizer, fols, printinfo=args.print_generate)
                log_file.write(f"Epoch {epoch+1}/{args.epochs}, Success Rate: {acc:.4f}\n")
                
                if acc > best_acc :
                    print('Best Accuracy updated : {} -> {}'.format(best_acc, acc))
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    with open(f"{save_dir}/args.json", 'w') as f:
                        json.dump(vars(args), f)
                    with open(f"{save_dir}/accuracy.txt", 'w') as f:
                        f.write(f"Test Accuracy: {acc:.2f}\n")
                        f.write(f"Epoch: {str(epoch + 1)}\n")
                    with open(f"{save_dir}/generation.txt", 'w') as f:
                        f.writelines(generation)
                    print('Best Model Saved')
                    best_acc = acc

# argparse
def arg_parse():
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, required=True)
    args.add_argument('--lr', type=float, default=5e-5)
    args.add_argument('--model_name', type=str, default='gpt2')
    args.add_argument('--load_dir', default='', type=str)
    args.add_argument('--print_generate', action='store_true')
    args.add_argument('--note', type=str, default='exp5')
    args.add_argument('--seed', type=int, default=42)

    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--start', default=0, type=int)

    args.add_argument('--eval_steps', default=10, type=int)
    args.add_argument('--save_dir', default='', type=str)
    args.add_argument('--patience', default=100, type=int)
    # args.add_argument('--accel', default=False, action='store_true')



    parsed = args.parse_args()
    parsed.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed

# main
if __name__ == "__main__":
    args = arg_parse()
    print(json.dumps(vars(args), indent=2))
    train(args)