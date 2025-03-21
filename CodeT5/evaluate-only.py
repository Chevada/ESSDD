import os
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
import numpy as np


def eval_result(output_fn,gold_fn,lang):
    # 初始化 dev_accs 列表
    dev_accs = []
    bleu, codebleu = 0.0, 0.0

    # 读取生成的结果
    with open(output_fn, 'r', encoding="utf-8") as f, open(gold_fn, 'r', encoding="utf-8") as f1:
        pred_nls = f.readlines()
        gold_targets = f1.readlines()

    # 计算准确率
    for pred_nl, gold in zip(pred_nls, gold_targets):
        # 比较预测结果和目标结果
        dev_accs.append(pred_nl.strip() == gold.strip())
        if(pred_nl.strip() == gold.strip()):
            print(f'此类匹配成功：{gold}')
    bleu = round(_bleu(gold_fn, output_fn), 2)
    codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, lang)
    result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
    result['codebleu'] = codebleu * 100
    print("***** Eval results *****")
    for key in sorted(result.keys()):
        print("  %s = %s", key, str(round(result[key], 4)))

    return result

if __name__ == '__main__':
    # output_fn = 'DeepSeek-V3_evaluation_results/test_output_filtered.txt'
    # gold_fn = 'DeepSeek-V3_evaluation_results/test_gold_filtered.txt'
    # output_fn = 'gpt-3.5-turbo_evaluation_results/test_output.txt'
    # gold_fn = 'gpt-3.5-turbo_evaluation_results/test_gold.txt'

    output_fn = 'E:/learing/论文复现/CodeT5/sh/saved_models/gen_class/codet5_base_all_lr3_bs24_src64_trg380_pat3_e40_20250217_0829/prediction/test_best-ppl.output'
    gold_fn = 'E:/learing/论文复现/CodeT5/sh/saved_models/gen_class/codet5_base_all_lr3_bs24_src64_trg380_pat3_e40_20250217_0829/prediction/test_best-ppl.gold'
    lang = 'python'
    eval_result(output_fn,gold_fn,lang)