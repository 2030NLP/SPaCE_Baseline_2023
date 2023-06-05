CUDA_VISIBLE_DEVICES=0 python ./src/task2/predict.py \
    --data_path ./data/input/task2 \
    --output_path ./data/model/task2 \
    --load_model_path ./data/model \
    --base_model hfl/chinese-bert-wwm-ext \
    --seq_max_length 256 \
    --split test \
    --eval_batch_size 32 \
    --seed 42 \
    --cuda 