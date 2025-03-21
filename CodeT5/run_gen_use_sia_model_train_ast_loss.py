
import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model,ASTLossPredictor
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data,filter_encoder_state_dict,weighted_edit_distance_loss
from configs import add_args, set_seed, set_dist
import pickle
from siamese_model import SiameseEncoder,ContrastiveLoss
from SPTCode.ast_parser import  generate_single_enhanced_ast
import random
# from transformers import (T5Config, T5ForConditionalGeneration, T5Tokenizer)

# 测试
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0


    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        # batch = tuple(t.to(device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss
#  添加此段，以适应多GPU训练
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        eval_loss += loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                # preds = model.generate(source_ids,
                preds = model.module.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if args.task in ['summarize']:
                    # for smooth-bleu4 evaluation
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                else:
                    # 将类内部的换行符替换为 '\n' 字符
                    formatted_pred_nl = pred_nl.replace('\n', '\\n').strip()
                    formatted_gold_target = gold.target.replace('\n', '\\n').strip()
                    formatted_gold_source = gold.source.replace('\n', '\\n').strip()
                    f.write(formatted_pred_nl + '\n')
                    f1.write(formatted_gold_target + '\n')
                    f2.write(formatted_gold_source + '\n')
                    # f.write(pred_nl.strip() + '\n')
                    # f1.write(gold.target.strip() + '\n')
                    # f2.write(gold.source.strip() + '\n')

        if args.task == 'summarize':
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if args.task in ['concode', 'translate', 'refine','gen_class','sia_gen']:
                codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)

        result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
        if args.task == 'concode' or args.task == 'gen_class' or args.task == 'sia_gen':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

def eval_result(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    args.res_dir = 'saved_models/gen_class/codet5_base_all_lr3_bs24_src64_trg350_pat3_e40_20241016_111224/prediction'
    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))
    # 初始化 dev_accs 列表
    dev_accs = []
    bleu, codebleu = 0.0, 0.0

    # 读取生成的结果
    with open(output_fn, 'r') as f, open(gold_fn, 'r') as f1:
        pred_nls = f.readlines()
        gold_targets = f1.readlines()

    # 计算准确率
    for pred_nl, gold in zip(pred_nls, gold_targets):
        # 比较预测结果和目标结果
        dev_accs.append(pred_nl.strip() == gold.strip())
    bleu = round(_bleu(gold_fn, output_fn), 2)
    if args.task in ['concode', 'translate', 'refine', 'gen_class', 'sia_gen']:
        codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)
    result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
    if args.task == 'concode' or args.task == 'gen_class' or args.task == 'sia_gen':
        result['codebleu'] = codebleu * 100
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result



def calculate_ast_loss(outputs,target_ids,tokenizer):
    # 改进点
    # 从 logits 中提取生成标记序列
    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=-1)  # 获取每个时间步的最高概率标记


    # 初始化用于保存生成的代码和目标代码
    generated_codes = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    target_codes = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

    # 批量生成 AST
    generated_asts = [generate_single_enhanced_ast(code, 'python').split(' ') for code in generated_codes]
    target_asts = [generate_single_enhanced_ast(code, 'python').split(' ') for code in target_codes]

    # ast_loss = 0.0
    # 初始化一个列表来保存每个样本的 AST 损失
    ast_losses = []

    # 遍历批次中的每个样本，计算 AST 序列和损失
    for gen_ast, tgt_ast in zip(generated_asts, target_asts):
        # 计算每个样本的 AST 损失并累加
        # ast_loss += weighted_edit_distance_loss(gen_ast, tgt_ast)
        sample_loss = weighted_edit_distance_loss(gen_ast, tgt_ast)
        ast_losses.append(sample_loss)

    # # 计算平均 AST 损失
    # ast_loss /= len(generated_asts)  # 对批次中的样本数进行平均
    # return ast_loss
    # 将损失列表转换为张量并返回
    return torch.tensor(ast_losses)  # 返回损失向量


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)

    # 使用 CodeT5 的 hidden size 初始化 ASTLossPredictor
    hidden_size = model.config.hidden_size  # 从 CodeT5 的配置中获取
    astLossModel = ASTLossPredictor(32100,hidden_size)


    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        astLossModel = torch.nn.DataParallel(astLossModel)
    model.to(args.device)
    astLossModel.to(args.device)

    # 实际使用的也是gen_class的数据
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, 'gen_class', args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:

        # 加载数据集
        data = torch.load('/AIsoftwaremfq2023/code/cl_code/CodeT5/data/ast_loss_data/train.pt')
        pooled_logits = data['pooled_logits']
        ast_loss = data['ast_loss']

        # 创建 TensorDataset
        dataset = TensorDataset(pooled_logits, ast_loss)

        # 创建 DataLoader
        train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

        # 定义 ASTLossPredictor 的优化器
        ast_optimizer = AdamW(astLossModel.parameters(), lr=0.001)

        mse_loss_fn = nn.MSELoss()

        # Start training
        train_example_num = len(dataset)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)


        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            fcn_loss_total=0 # 初始化累计损失
            astLossModel.train()

            for step, batch in enumerate(bar):
                logits, ast_loss = batch
                logits = logits.to(args.device)
                ast_loss = ast_loss.to(args.device)

                predicted_ast_loss = astLossModel(logits)
                predicted_ast_loss = predicted_ast_loss.squeeze()  # 去掉维度为1的维度

                # 定义放大系数
                # scaling_factor = 10.0  # 你可以根据需要调整这个值

                # 同时放大
                # scaled_predicted_ast_loss = predicted_ast_loss * scaling_factor
                # scaled_ast_loss = ast_loss * scaling_factor

                fcn_loss = mse_loss_fn(predicted_ast_loss, ast_loss)

                # 记录 fcn_loss
                fcn_loss_total += fcn_loss.item()  # 累计 fcn_loss

                ast_optimizer.zero_grad()
                fcn_loss.backward()
                ast_optimizer.step()
                bar.set_description("[{}] FCN loss {}".format(
                    cur_epoch,round(fcn_loss_total / (step + 1), 3) if step > 0 else 0
                ))

                if fcn_loss_total / (step + 1) < 0.01 or cur_epoch > 10:
                    logger.info("Stopping training at epoch {} with FCN loss {}".format(cur_epoch, fcn_loss))
                    # 保存模型参数
                    torch.save(astLossModel.state_dict(), '/AIsoftwaremfq2023/code/cl_code/CodeT5/sh/saved_models/ast_loss/final_astLossModel.bin')
                    break  # 结束当前轮次的训练

            if args.do_eval:
                pass
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        pass
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
