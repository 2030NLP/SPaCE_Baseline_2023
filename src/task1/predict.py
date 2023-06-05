import os
import argparse
import torch
import json
import random
import time
import numpy as np

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer

from model import Task1Model
import data
import utils


def load_model(params, base_path):
    full_path = os.path.join(base_path, 'checkpoint.bin')
    params['load_model_path'] = full_path
    model = Task1Model(params)
    return model


def predict(
    model, eval_dataloader, params, device, logger,
):
    with torch.no_grad():
        model.eval()
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

        results = []
    
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, attention_mask = batch
            
            # predict tags
            tag_prediction = model.eval().predict(
                input_ids, 
                token_type_ids, 
                attention_mask,
            )

            tag_prediction = tag_prediction.detach().cpu().numpy()
            results.append(tag_prediction)

        results = np.concatenate(results, axis=0)
        return results


def gather_segments(input_text, predicted_tags):
    all_segments = []

    num_tags = 6
    indices = [[] for i in range(num_tags)]
    names = ['S1', 'P1', 'E1', 'S2', 'P2', 'E2']
    for i, x in enumerate(predicted_tags):
        if (x >= 1) and (x <= 6):
            indices[x-1].append(i-1)
    texts = [
        ''.join([input_text[i] for i in indices[j]]) for j in range(num_tags)
    ]
    segment = []
    for i in range(num_tags):
        segment.append({
            'role': names[i],
            'text': texts[i],
            'idxes': indices[i],
        })
    all_segments.append(segment)

    return all_segments


def main(params):
    model_output_path = params['output_path']
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    logger = utils.setup_logger('SpaCETask1', params['output_path'])
    logger.info("Evaluation on the %s set." %params['split'])
    tokenizer = AutoTokenizer.from_pretrained(params['base_model'])

    # Init model
    base_model_path = params['load_model_path']
    model = load_model(params, base_model_path)
    device = model.device
    model.to(device)


    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load eval data
    test_samples, test_num = data.read_split(params["data_path"], params['split'], test_mode=True)
    if (logger):
        logger.info("Read %d test samples." % test_num)

    test_tensor_data = data.process_data(
        test_samples,
        tokenizer,
        params,
        test_mode=True,
    )
    test_sampler = SequentialSampler(test_tensor_data)

    test_dataloader = DataLoader(
        test_tensor_data, sampler=test_sampler, batch_size=params['eval_batch_size']
    )

    # evaluate
    tag_results = predict(
        model, test_dataloader, params, device=device, logger=logger,
    )

    # print(len(test_samples))
    # print(len(tag_results))
    
    out_file_path = os.path.join(params['output_path'], '%s_prediction.jsonl' %params['split'])
    with open(out_file_path, 'w', encoding='utf-8') as fout:
        for i, js in enumerate(test_samples):
            predicted_tags = tag_results[i]
            output = gather_segments(
                js['context'], 
                predicted_tags,
            )
            out_js = {
                'qid': js['qid'],
                'results': output,
            }
            json.dump(out_js, fout, ensure_ascii=False)
            fout.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--data_path', type=str, default='./data/raw/task1')
    parser.add_argument('--output_path', type=str, default='./data/model/task1')
    parser.add_argument('--load_model_path', type=str, default='./data/model/task1/checkpoint.bin')
    parser.add_argument('--base_model', type=str, default='hfl/chinese-bert-wwm-ext')
    
    # model arguments
    parser.add_argument('--seq_max_length', type=int, default=256)
    parser.add_argument('--cuda', action='store_true')

    # evaluation arguments
    parser.add_argument('--split', type=str, default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    main(params)