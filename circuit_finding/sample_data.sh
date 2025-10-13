CUDA_VISIBLE_DEVICES=1 python sample_w_loss_peacok.py --model EleutherAI/gpt-j-6B \
--data_name ../data/peacok_person_t1_name \
--model_dir ../finetuned_models/gpt-j-6B_t1_chain_name_gptj_full \
--out_dir sample/peacok_gptj_t1_sample


CUDA_VISIBLE_DEVICES=1 python sample_w_loss_peacok.py --model Qwen/Qwen2.5-7B-Instruct \
--data_name ../data/peacok_person_t1_name \
--model_dir ../finetuned_models/Qwen2.5-7B-Instruct_t1_chain_name_qweninst_full \
--out_dir sample/peacok_qweninst_t1_sample