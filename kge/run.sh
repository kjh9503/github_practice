CUDA_VISIBLE_DEVICES=2 python -u ./codes/run.py \
    --do_train \
    --do_test \
    --cuda \
    --data_path data/peacok_person_t1_name \
    --model TransE \
    -n 128 -b 64 -d 4096 \
    -adv \
    --max_steps 10000 \
    -save ./log/peacok_person_t1_name --test_batch_size 16 \
    --test_batch_size 64 
    #--evaluate_train

CUDA_VISIBLE_DEVICES=2 python -u ./codes/run.py \
    --do_train \
    --do_test \
    --cuda \
    --data_path data/peacok_person_t1.2_name \
    --model TransE \
    -n 128 -b 64 -d 4096 \
    -adv \
    --max_steps 10000 \
    -save ./log/peacok_person_t1.2_name --test_batch_size 16 \
    --test_batch_size 64 
    #--evaluate_train


