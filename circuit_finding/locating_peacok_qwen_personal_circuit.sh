CUDA_VISIBLE_DEVICES=3 python locating_qweninst_all_person.py \
--save_path ./results/peacok_nlq_qweninst_t1_person \
--sample_dir ./sample/peacok_nlq_qweninst_t1_person --batch_size 1 \
--limit 822 \
--model_dir ../finetuned_models/Qwen2.5-7B-Instruct_t1_chain_name_qweninst_full \
--head_num 8 \
--chosen_layers '0-27'