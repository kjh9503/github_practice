import re
import torch
from bert_score import score
import numpy as np
from tqdm import trange
from transformers import logging
logging.set_verbosity_error()

def parse_log_and_evaluate(log_text: str):
    cases = log_text.strip().split("-----------------------------------------------------------")
    parsed = []

    for case in cases:
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
        parsed.append((ground_truth.strip(), llm_generation))

    references = [gt for gt, _ in parsed]
    candidates = [gen for _, gen in parsed]

    # BERTScore 계산
    P, R, F1 = score(candidates, references, lang="en")
    return F1


# dir_name = './log/batch_chain_essence_0.1_locality_L0_N16_E18/locality_log'
# dir_name = './log/batch_chain_essence_0.0625_locality_L0_N16_E18/locality_log'
# dir_name = './log/batch_chain_essence_0.01_locality_L0_N16_E18/locality_log'
# dir_name = './log/batch_chain_locality_L0_N16_E15/locality_log'
# start, end, inter = 0, 52, 1

# dir_name = './log/sel_tune_locality_layer_0'
dir_name = './log/sel_tune_locality_layer_27'
start, end, inter = 0, 832, 16

scores = []
for i in trange(start, end, inter):
    with open(f'{dir_name}/locality_generation_batch_num_{i}.txt', 'r') as f :
        log = f.read()
    F1 = parse_log_and_evaluate(log)
    score_ = torch.mean(F1)
    scores.append(score_.item())

if 'essence_0.1' in dir_name :
    case_name = 'essence_0.1'
elif 'essence_0.0625' in dir_name :
    case_name = 'essence_0.0625'
elif 'essence_0.01' in dir_name :
    case_name = 'essence_0.01'
elif 'batch_chain_locality_L0_N16_E15' in dir_name :
    case_name = 'no essence'
elif 'sel_tune_locality_layer_0' in dir_name :
    case_name = 'sel_tune_l0'
elif 'sel_tune_locality_layer_27' in dir_name :
    case_name = 'sel_tune_l27'
else :
    raise ValueError('Invalid exp name')

print(case_name, ':',  np.mean(scores))
