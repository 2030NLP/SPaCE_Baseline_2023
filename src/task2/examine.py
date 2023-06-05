import os
import argparse
import json
import numpy as np
import scipy
import traceback


name_to_index = {
    '空间实体': 1,
    '参照实体': 2,
    '事件': 3,
    '事实性': 4,
    '时间': 5,
    '处所': 6,
    '起点': 7,
    '终点': 8,
    '方向': 9,
    '朝向': 10,
    '部件处所': 11,
    '部位': 12,
    '形状': 13,
    '路径': 14,
    '距离': 15,
}


def intersection_and_union(input, target):
    _input, _target = set(input), set(target) 
    intersection = _input & _target
    union = _input | _target

    return len(intersection), len(union)


def cal_similarity(golden_tuple, predicted_tuple, corefs, params):
    # if (len(golden_tuple) != len(predicted_tuple)):
    #     return 0

    non_null_pair = 0
    total_score = 0.0

    golden_map, predicted_map = {}, {}
    for element in golden_tuple:
        golden_map[element['role']] = element
    for element in predicted_tuple:
        predicted_map[element['role']] = element

    for element_name in name_to_index:
        if (element_name not in golden_map) and (element_name not in predicted_map):
            continue

        non_null_pair += 1
        if (element_name not in golden_map) or (element_name not in predicted_map):
            element_sim_score = 0
        else:
            g_element, p_element = golden_map[element_name], predicted_map[element_name]
            if ('label' in g_element): # 标签类元素
                if not('label' in p_element):
                    element_sim_score = 0.0
                elif (g_element['label'] != p_element['label']):
                    element_sim_score = 0.0
                else:
                    element_sim_score = 1.0
            else: # 原文片段类元素
                p_idx, g_idx = p_element['fragment']['idxes'], g_element['fragment']['idxes']
                p_text, g_text = p_element['fragment']['text'], g_element['fragment']['text']

                # 计算原文本的相似度
                n_inter, n_union = intersection_and_union(p_text, g_text)
                element_sim_score = n_inter/n_union

                if ((element_name == '空间实体') or (element_name == '参照实体')): # 如果是空间实体，使用idx评价
                    n_inter, n_union = intersection_and_union(p_idx, g_idx)
                    element_sim_score = n_inter/n_union

                    # 尝试取所有共指中重合度最高的一个
                    g_idx_set = set(g_idx)
                    for key in corefs:
                        key_idx_set = set(eval(key))
                        if (key_idx_set.issubset(g_idx_set)):
                            diff_set = g_idx_set - key_idx_set
                            for c in corefs[key]:
                                corefed_g_idx = set(c['idxes']) | diff_set
                                n_inter, n_union = intersection_and_union(p_idx, corefed_g_idx)
                                element_sim_score = max(element_sim_score, n_inter/n_union)
                    
                    if (params['debug']):
                        print('Golden entity: ', g_element)
                        print('Predicted entity: ', p_element)
                        print('Score: ', element_sim_score)
                        input()

        if ((element_name == '空间实体') or (element_name == '参照实体')) and (element_sim_score == 0): # 关键实体（空间实体）不能完全错误
            return 0

        total_score += element_sim_score

    return total_score/non_null_pair


def KM_algorithm(pair_scores):
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-pair_scores) # 求负将最大和转变为最小和
    max_score = pair_scores[row_ind, col_ind].sum()
    return max_score


def main(params):
    answers = {}
    with open(params['answer_path'], 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            answers[js['qid']] = js

    predictions = {}
    with open(params['prediction_path'], 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            if ('qid' in js):
                predictions[js['qid']] = js

    precisions, recalls, f1s = [], [], []
    for qid in answers:
        if (qid not in predictions):
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)
        else:
            x, y = answers[qid], predictions[qid]    
            
            if (params['debug']):
                print(x['context'])
                print(x['corefs'])
            
            # build coreference set
            corefs = {}
            for coref_set in x['corefs']:
                for coref_element in coref_set:
                    idx_str = str(coref_element['idxes'])
                    if (idx_str not in corefs):
                        corefs[idx_str] = coref_set
            
            golden_outputs = x['results']
            M = len(golden_outputs)
            predicted_outputs = y['results']
            N = len(predicted_outputs)
            if (N > 100): # malicious submit
                continue

            pair_scores = np.zeros((M, N))
            for i in range(M):
                for j in range(N):
                    pair_scores[i][j] = cal_similarity(
                        golden_outputs[i],
                        predicted_outputs[j],
                        corefs,
                        params,
                    )

            max_bipartite_score = KM_algorithm(pair_scores)
            if (N == 0):
                _precision = 0
            else:
                _precision = max_bipartite_score/N
            if (M == 0):
                _recall = 0
            else:
                _recall = max_bipartite_score/M
            if (_precision+_recall == 0):
                _f1 = 0
            else:
                _f1 = 2*(_precision*_recall)/(_precision+_recall)
            precisions.append(_precision)
            recalls.append(_recall)
            f1s.append(_f1)

        status = 'Accepted'
        avg_precision = sum(precisions)/len(answers)
        avg_recall = sum(recalls)/len(answers)
        if (avg_precision+avg_recall == 0):
            micro_f1 = 0
        else:
            micro_f1 = 2*(avg_precision*avg_recall)/(avg_precision+avg_recall)
        macro_f1 = sum(f1s)/len(answers)

        final_result = {
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
        }

    if (params['debug']):
        print(status)
        if (final_result is not None):
            print('Micro F1 score: %f' %(final_result['micro_f1']))
            print('Macro F1 score: %f' %(final_result['macro_f1']))
            print('Average precision: %f' %(final_result['avg_precision']))
            print('Average recall: %f' %(final_result['avg_recall']))

    return status, final_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_path', type=str, default='./data/input/task2/task2_dev.jsonl')
    parser.add_argument('--prediction_path', type=str, default='./data/input/task2/task2_dev.jsonl')
    parser.add_argument('--output_path', type=str, default='./data/scores/task2')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    params = args.__dict__
    print(params)
    
    try:
        status, final_result = main(params)
    except:
        traceback.print_exc()
        status, final_result = 'Error in execution', None

    print(status)
    if (final_result is not None):
        print(json.dumps(final_result, indent=2))

    if not(os.path.exists(params['output_path'])):
        os.makedirs(params['output_path'])

    with open(os.path.join(params['output_path'], 'score.json'), 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(final_result, indent=4))