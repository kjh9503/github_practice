

CUDA_VISIBLE_DEVICES=3 python locating_gptj.py \
--save_path ./results/peacok_gptj_nlq_t1_chain_1p \
--sample_dir ../circuit_finding/sample/peacok_nlq_gptj_t1_chain_1p --batch_size 128 \
--limit 822 \
--model_dir ../finetuned_models/gpt-j-6B_t1_chain_name_gptj_full \
--head_num 8 \
--chosen_layers '0-27'
