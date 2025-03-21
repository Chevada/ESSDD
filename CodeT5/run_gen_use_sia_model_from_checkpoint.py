
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
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
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

    params_path = 'saved_models/sia_gen/codet5_base_all_useeuc_lr3_bs24_src64_trg380_pat3_e40_20241021_023430'
    params_file = os.path.join(params_path, 'checkpoint-{}/pytorch_model.bin'.format('best-ppl'))

    # model.load_state_dict(torch.load(file))

    # encoder_params_path = 'saved_models/siamese/codet5_base_all_useeuc_lr10_bs40_src64_trg150_pat2_e30_20241021_002115'
    # encoder_params_file = os.path.join(encoder_params_path, 'checkpoint-{}/pytorch_model.bin'.format('best-ppl'))
    # pretrained_encoder_state_dict = filter_encoder_state_dict(torch.load(encoder_params_file))

    ast_loss_param_path = 'saved_models/ast_loss/final_astLossModel.bin'

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        astLossModel = torch.nn.DataParallel(astLossModel)
    model.to(args.device)

    astLossModel.to(args.device)
    # model.module.load_state_dict(torch.load(encoder_params_file))
    # model.module.encoder.load_state_dict(pretrained_encoder_state_dict)
    model.module.load_state_dict(torch.load(params_file))
    astLossModel.load_state_dict(torch.load(ast_loss_param_path))

    # logger.info(f"Successfully imported parameters from siamese network {encoder_params_path}")s
    logger.info(f"Successfully imported parameters from sia_gen network {params_path}")


    pool = multiprocessing.Pool(args.cpu_cont)
    # 实际使用的也是gen_class的数据
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, 'gen_class', args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        # no_decay定义了不进行权重衰减（weight decay）的参数，这是一种正则化的方法，用于防止模型过拟合
        no_decay = ['bias', 'LayerNorm.weight']
        # 将模型的参数分为两个字典，分别为需要更新的和不需要更新的
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        # 学习率预热，是针对学习率learning rate优化的一种策略，
        # 主要过程是，在预热期间，学习率从0线性（也可非线性）增加到优化器中的初始预设lr，之后使其学习率从优化器中的初始lr线性降低到0
        # 参数说明： optimizer： 优化器 num_warmup_steps：初始预热步数 num_training_steps：整个训练过程的总步数
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # 此处冻结astLossModel的参数，损失反向传播到外层的codet5模型中，现在此处进行一次反向传播，之后还会有一次根据python类的交叉熵损失的反向传播
        # 冻结 astLossModel 的参数
        for param in astLossModel.parameters():
            param.requires_grad = False

        # 定义两个模型的联合优化器，目的是让astLossModel的输出可以反向传播，分别经过astLossModel和codet5模型
        # 将 model 和 astLossModel 的参数组合到一起
        # optimizer_union_grouped_parameters = [
        #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #      'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #      'weight_decay': 0.0},
        #     {'params': astLossModel.parameters(), 'weight_decay': args.weight_decay}  # 添加 astLossModel 的参数
        # ]
        #
        # union_optimizer = AdamW(optimizer_union_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        #
        # mse_loss_fn = nn.MSELoss()

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6
        code_weight = 0.5
        ast_weight = 0.5

        # 此处是为了记录整个过程中的损失和ppl变化情况
        loss_list = []
        ppl_list = []

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            train_loss,fcn_loss_total, predicted_ast_loss_total =0, 0, 0  # 初始化累计损失
            model.train()
            astLossModel.train()

            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                    pooled_logits = torch.mean(outputs.logits, dim=1)

                    predicted_ast_loss = astLossModel(pooled_logits)
                    predicted_ast_loss = predicted_ast_loss.mean()

                    # 记录 predicted_ast_loss
                    predicted_ast_loss_total += predicted_ast_loss.item()  # 累计 predicted_ast_loss
                    # predicted_ast_loss = predicted_ast_loss*ast_weight

                    # 反向传播 预测的predicted_ast_loss，通过 astLossModel 和 CodeT5，使用预测的ast损失去训练codet5的参数，目的是使ast损失最小化
                    predicted_ast_loss.backward(retain_graph=True)  # 允许后续的反向传播

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    tr_loss += loss.item()

                    # 用于记录训练过程中已处理的样本数量
                    nb_tr_examples += source_ids.size(0)
                    nb_tr_steps += 1
                    # loss = loss*code_weight
                    loss.backward()

                    if nb_tr_steps % args.gradient_accumulation_steps == 0:
                        # Update parameters
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        global_step += 1
                        train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}, Predicted AST loss {}".format(
                        cur_epoch, round(train_loss, 3),
                        round(predicted_ast_loss_total / (step + 1), 3) if step > 0 else 0
                    ))
            # 记录当前轮次的最终损失
            # epoch_train_loss = round(tr_loss / nb_tr_steps, 4)
            # loss_list.append(epoch_train_loss)

            if args.do_eval:
                # Eval model with dev data
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                ppl_list.append(eval_ppl)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

                # 以下内容有待实现
                if args.do_eval_bleu:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                       only_src=True, is_sample=True)

                    result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                    dev_bleu, dev_em = result['bleu'], result['em']
                    if args.task in ['summarize']:
                        dev_bleu_em = dev_bleu
                    elif args.task in ['defect']:
                        dev_bleu_em = dev_em
                    else:
                        dev_bleu_em = dev_bleu + dev_em
                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                        # tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best bleu model into %s", output_model_file)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        fa.write(
                            "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f (bleu: %.2f, em: %.2f)\n" % (
                                cur_epoch, best_bleu_em, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em))
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

        loss_output_dir = os.path.join(args.output_dir, 'loss_list.pkl')
        ppl_output_dir = os.path.join(args.output_dir, 'ppl_list.pkl')
        # 保存损失列表到文件中
        with open(loss_output_dir, 'wb') as f:
            pickle.dump(loss_list, f)
        # 保存ppl列表到文件中
        with open(ppl_output_dir, 'wb') as f:
            pickle.dump(ppl_list, f)

    if args.do_test:

        args.eval_batch_size = 8

        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        # for criteria in ['best-bleu']:
        # for criteria in ['best-ppl']:
        # 单独测试时，需要单独指定output_dir以让模型导入参数
        args.output_dir = 'saved_models/sia_gen/codet5_base_all_useeuc_lr3_bs24_src64_trg380_pat3_e40_20241030'
        # for criteria in ['best-ppl']:
        for criteria in ['last']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))

            # model.load_state_dict(torch.load(file))
            model.module.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
            # 不需要重新生成
            # result = eval_result(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
            test_bleu, test_em = result['bleu'], result['em']
            test_codebleu = result['codebleu'] if 'codebleu' in result else 0
            result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (criteria, test_bleu, test_em, test_codebleu)
            logger.info(result_str)
            fa.write(result_str)
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write(result_str)
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
