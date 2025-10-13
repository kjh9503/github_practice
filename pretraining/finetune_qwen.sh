CUDA_VISIBLE_DEVICES=3 python finetuning_llm_left.py \
--data_path ../data/peacok_person_t1_name \
--model_name 'Qwen/Qwen2.5-7B-Instruct' \
--seed 42 \
--epochs 100 \
--note t1_chain_name_qweninst_full \
--eval_steps 5 \
--patience 5 \
--batch_size 16 \
--lr 1e-5 