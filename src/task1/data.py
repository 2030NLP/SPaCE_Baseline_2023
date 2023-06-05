import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def read_split(data_path, split, test_mode=False):
    data = []
    if not(test_mode):
        file_name = 'task1_%s.jsonl' %(split)
    else:
        file_name = 'task1_%s_input.jsonl' %(split)

    with open(os.path.join(data_path, file_name)) as fin:
        for line in fin:
            js = json.loads(line)
            data.append(js)

            if not(test_mode) and (('results' not in js) or (js['results'] == [])):
                print(line)
                input()

    return data, len(data)


def process_data(data, tokenizer, params, test_mode=False):
    all_inputs = []
    all_outputs = []

    for js in data:
        text = js['context']
        if not(test_mode):
            outputs = js['results']

        tokenize_result = tokenizer(
            text=text,
            add_special_tokens=True,
            padding='max_length',
            max_length=params['seq_max_length'],
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_tensors='np',
        )
        input_ids, token_type_ids, attention_mask = tokenize_result['input_ids'][0], tokenize_result['token_type_ids'][0], tokenize_result['attention_mask'][0]
        offset_mapping = tokenize_result['offset_mapping'][0]

        # ids_to_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        # print(ids_to_tokens)
        # input()

        if not(test_mode):
            for segments in outputs:
                tag_labels = np.zeros(input_ids.shape, dtype=np.int32)
                tags = {
                    'S1': 1,
                    'P1': 2,
                    'E1': 3,
                    'S2': 4,
                    'P2': 5,
                    'E2': 6,
                }
                for t in segments:
                    indices = t['idxes']
                    for ind in indices:
                        for j in range(input_ids.shape[0]):
                            token_start, token_end = offset_mapping[j]
                            if (token_start == token_end): # special tokens
                                continue

                            if (token_start == ind):
                                tag_labels[j] = tags[t['role']]

                all_outputs.append([input_ids, token_type_ids, attention_mask, tag_labels])

        all_inputs.append([input_ids, token_type_ids, attention_mask])

    input_ids = torch.tensor(np.array([x[0] for x in all_inputs]), dtype=torch.long)
    token_type_ids = torch.tensor(np.array([x[1] for x in all_inputs]), dtype=torch.long)
    attention_mask = torch.tensor(np.array([x[2] for x in all_inputs]), dtype=torch.long)
    if not(test_mode):
        _input_ids = torch.tensor(np.array([x[0] for x in all_outputs]), dtype=torch.long)
        _token_type_ids = torch.tensor(np.array([x[1] for x in all_outputs]), dtype=torch.long)
        _attention_mask = torch.tensor(np.array([x[2] for x in all_outputs]), dtype=torch.long)
        tag_labels = torch.tensor(np.array([x[3] for x in all_outputs]), dtype=torch.long)
        tensor_dataset = TensorDataset(_input_ids, _token_type_ids, _attention_mask, tag_labels)
        
        return tensor_dataset
    else:
        tensor_dataset = TensorDataset(input_ids, token_type_ids, attention_mask)
        return tensor_dataset
    

if __name__ == '__main__':
    pass