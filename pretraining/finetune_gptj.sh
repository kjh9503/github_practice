CUDA_VISIBLE_DEVICES=2 python finetuning_llm_left.py \
--data_path ../data/peacok_person_t1_name \
--model_name 'EleutherAI/gpt-j-6B' \
--seed 42 \
--epochs 100 \
--note gpt-j-6B_t1_chain_name_gptj_full \
--eval_steps 5 \
--patience 5 \
--batch_size 32 \
--lr 1e-5 