CUDA_VISIBLE_DEVICES=1 python locating_gptj_all_person.py \
--save_path ./results/peacok_nlq_gptj_t1_chain_1p_person \
--sample_dir ../circuit_finding/sample/peacok_nlq_gptj_t1_chain_1p_person --batch_size 1 \
--limit 822 \
--model_dir ../finetuned_models/gpt-j-6B_t1_chain_name_gptj_full \
--head_num 8 \
--chosen_layers '0-27'