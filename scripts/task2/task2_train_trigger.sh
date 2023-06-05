CUDA_VISIBLE_DEVICES=0 python ./src/task2/train_trigger.py \
    --data_path ./data/input/task2 \
    --output_path ./data/model/task2_trigger \
    --base_model hfl/chinese-bert-wwm-ext \
    --seq_max_length 256 \
    --learning_rate 1e-5 \
    --epoch 4 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --print_interval 20 \
    --eval_interval 100 \
    --shuffle \
    --do_evaluate \
    --final_evaluate \
    --seed 42 \
    --cuda 