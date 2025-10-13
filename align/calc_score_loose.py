import re
import torch
from bert_score import score
import numpy as np
from tqdm import trange
import json

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
    
with open('../dataprocessing/peacok_person_t1_name/locality_ans.json', 'r') as f :
    loc_ans = json.load(f)

def parse_log_and_evaluate(log_text: str):
    cases = log_text.strip().split("-----------------------------------------------------------")

    cor_or_not = []

    for case in cases:
        cor = False
        lines = case.strip().splitlines()
        if not lines or not lines[0].startswith("["):
            continue

        try:
            prefix, first_gen_line = lines[0].split("] →", 1)
            prefix = prefix.lstrip("[")
            prompt, ground_truth = prefix.split(" | ", 1)
        except ValueError:
            continue
        
        # first_gen_line 에서 prompt 부분을 제거해줘야 함!
        prompt_prefix = prompt.strip()
        
        gt_cands = loc_ans[prompt_prefix]

        llm_gen1 = first_gen_line.strip()
        if prompt_prefix in llm_gen1:
            llm_gen1 = llm_gen1.replace(prompt_prefix, "").strip()

        # 나머지 줄까지 포함해서 generation 이어붙이기
        gen_lines = [llm_gen1]
        for line in lines[1:]:
            if line.strip().startswith("-------->"):
                break
            gen_lines.append(line.strip())

        llm_generation = " ".join(gen_lines).strip()

        
        for gt in gt_cands :
            cor = is_correct(llm_generation, gt)
            if cor :
                break
            cor = is_correct(llm_generation, ground_truth)
            if cor :
                break
        cor_or_not.append(cor)


    return cor_or_not, len(cases)


scores = []

# with open(f'./log/batch_chain_locality_L0_N16_E15/locality_log/locality_generation_batch_num_0.txt', 'r') as f :
#     log = f.read()
dir_name = './log/batch_chain_essence_0.01_locality_L0_N16_E18/locality_log'
# dir_name = './log/sel_tune_locality_layer_27'
start, end, inter = 0, 52, 1
# start, end, inter = 0, 832, 16



with open(f'{dir_name}/locality_generation_batch_num_0.txt', 'r') as f :
    log = f.read()
cases = log.strip().split("-----------------------------------------------------------")
tot = len(cases)


for i in trange(start, end, inter):
    with open(f'{dir_name}/locality_generation_batch_num_{i}.txt', 'r') as f :
        log = f.read()
    
    cor_or_not, tot_ = parse_log_and_evaluate(log)
    assert tot_ == tot
    score = np.mean(cor_or_not)
    scores.append(score)



print('total_score : ', np.mean(scores))
