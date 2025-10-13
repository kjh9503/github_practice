

CUDA_VISIBLE_DEVICES=3 python locating_qweninst.py \
--save_path ./results/peacok_qweninst_common \
--sample_dir sample/REIMPLE_peacok_qweninst_t1_sample \
--batch_size 32 \
--limit 822 \
--model_dir ../finetuned_models/Qwen2.5-7B-Instruct_t1_chain_name_qweninst_full \
--head_num 8 \
--chosen_layers '0-27'

