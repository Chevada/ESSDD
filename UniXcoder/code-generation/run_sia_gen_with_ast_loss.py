# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import sys
from bleu import _bleu
import pickle
import torch
import time
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq,ASTLossPredictor
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import filter_encoder_state_dict,get_elapse_time,filter_Roberta_encoder_state_dict,read_concode_examples

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

t0 = time.time()


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            # examples中的每个元素都是一个read_examples的类对象，有着idx。source，target这三个属性。
            examples.append(
                Example(
                    idx=idx,
                    source=" ".join(js['nl'].split()),
                    target=" ".join(js["code"].split()),
                )
            )

    return examples


def read_pyclass_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["input"].strip(),
                    target=x["label"].strip()
                )
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    # features列表中每个元素都是inputfeatures类的一个实例化对象，有着example_id，source_ids，target_ids三个属性，后两个都是序列词汇索引值
    features = []
    for example_index, example in enumerate(examples):
        # source
        # 源文本序列
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 5]
        # 源文本序列中插入特殊的序列
        source_tokens = [tokenizer.cls_token, "<encoder-decoder>", tokenizer.sep_token] + source_tokens + ["<mask0>",
                                                                                                           tokenizer.sep_token]
        # 序列转化为索引
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.max_source_length - len(source_ids)
        # 填充序列
        source_ids += [tokenizer.pad_token_id] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length

        # if example_index < 5:
        #     if stage=='train':
        #         logger.info("*** Example ***")
        #         logger.info("idx: {}".format(example.idx))

        # logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
        # logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
        #
        # logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
        # logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
            )
        )
    return features


def set_seed(seed=42):
    random.seed(seed)
    # 环境变量，用于设置系统的哈希种子
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    # action，action=‘store_true’/‘store_false’。使用这个选项的参数必须为布尔变量。其中store_true表示：用户指定了这个参数，那么这个参数就为true
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    # --weight_decay 是一个命令行参数，用于控制权重衰减（weight decay）的值。权重衰减是在训练神经网络时一种常用的正则化技巧，用于防止模型过拟合。
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--lang', default='java', type=str, help="train data language")

    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    # import！！！you must set is_decoder as True for generation
    config.is_decoder = True
    encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)

    model = Seq2Seq(encoder=encoder, decoder=encoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0], eos_id=tokenizer.sep_token_id)

    astLossModel = ASTLossPredictor(51416, 768)

    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)
    astLossModel.to(args.device)

    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)
        astLossModel = torch.nn.DataParallel(astLossModel)


    # 导入孪生网络的参数
    params_path = 'saved_models/concode/sia_gen'
    # params_path = 'saved_models/gen_class'
    params_file = os.path.join(params_path, 'checkpoint-{}/pytorch_model.bin'.format('last'))
    # params_file = os.path.join(params_path, 'checkpoint-{}/pytorch_model.bin'.format('best-ppl'))

    ast_loss_param_path = 'saved_models/concode/ast_loss/checkpoint-last/pytorch_model.bin'

    model.module.load_state_dict(torch.load(params_file))
    astLossModel.load_state_dict(torch.load(ast_loss_param_path))

    logger.info(f"Successfully imported parameters from sia_gen network {params_path}")

    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        # Prepare training data loader
        # train_examples = read_pyclass_examples(args.train_filename)
        train_examples = read_concode_examples(args.train_filename)
        # features列表中每个元素都是inputfeatures类的一个实例化对象，有着example_id，source_ids，target_ids三个属性，后两个都是序列词汇索引值
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        # 所有源序列的索引列表
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        # torch.utils.data.tensordataset
        train_data = TensorDataset(all_source_ids, all_target_ids)
        train_sampler = RandomSampler(train_data)
        # args.train_batch_size // args.gradient_accumulation_steps 就是计算在梯度累积过程中每次实际处理的小批量数据大小
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        # Prepare optimizer and schedule (linear warmup and decay)
        # 这段代码用于准备优化器和学习率调度器，其中包括权重衰减（weight decay）和线性预热和衰减（linear warmup and decay）的设置
        # 通常在优化器中，偏置（bias）和归一化层（LayerNorm）的权重参数不需要进行权重衰减。
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        # get_linear_schedule_with_warmup是一个函数，用于生成一个线性预热和衰减的学习率调度器。预热是指在训练初期逐渐增加学习率，
        # 以帮助模型更快地收敛。衰减是指在训练后期逐渐减小学习率，以细化模型的参数调整
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(
                                                        len(train_dataloader) * args.num_train_epochs * 0.1),
                                                    num_training_steps=len(train_dataloader) * args.num_train_epochs)

        # 冻结 astLossModel 的参数
        for param in astLossModel.parameters():
            param.requires_grad = False

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        # model.train()
        patience, best_score, losses, dev_dataset = 3, 0, [], {}
        not_loss_dec_cnt = 0
        global_step,best_ppl = 0,1e6
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            train_loss, fcn_loss_total, predicted_ast_loss_total = 0, 0, 0  # 初始化累计损失
            model.train()
            for idx, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids, target_ids = batch
                loss, _, _,outputs = model(source_ids=source_ids, target_ids=target_ids)

                pooled_logits = torch.mean(outputs, dim=1)

                predicted_ast_loss = astLossModel(pooled_logits)
                predicted_ast_loss = predicted_ast_loss.mean()

                # 记录 predicted_ast_loss
                predicted_ast_loss_total += predicted_ast_loss.item()  # 累计 predicted_ast_loss

                predicted_ast_loss.backward(retain_graph=True)  # 允许后续的反向传播

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_steps += 1

                # losses.append(loss.item())
                # 表示损失反向传播记录，但并没有更新
                loss.backward()
                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    # if len(losses) // args.gradient_accumulation_steps % 100 == 0:
                    #     logger.info("epoch {} step {} loss {}".format(epoch,
                    #                                                   len(losses) // args.gradient_accumulation_steps,
                    #                                                   round(np.mean(losses[
                    #                                                                 -100 * args.gradient_accumulation_steps:]),
                    #                                                         4)))
                bar.set_description("[{}] Train loss {}, Predicted AST loss {}".format(
                    epoch, round(train_loss, 3),
                    round(predicted_ast_loss_total / (idx + 1), 3) if idx > 0 else 0
                ))

            # 每个epoch训练之后便进行验证
            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    # eval_examples = read_examples(args.dev_filename)
                    # eval_examples = read_pyclass_examples(args.dev_filename)
                    eval_examples = read_concode_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_target_ids)
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # Start Evaling model
                model.eval()
                eval_loss, tokens_num = 0, 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, target_ids = batch

                    with torch.no_grad():
                        _, loss, num,_ = model(source_ids=source_ids, target_ids=target_ids)
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()

                    # tokens_num += num.sum().item()
                # Pring loss of dev dataset
                eval_loss = eval_loss / tokens_num
                eval_ppl = round(np.exp(eval_loss), 5)
                result = {'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info("Save the last model into %s", output_model_file)

                if epoch>=10:
                    each_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(each_output_dir):
                        os.makedirs(each_output_dir)
                    each_model_to_save = model.module if hasattr(model, 'module') else model
                    each_output_model_file = os.path.join(each_output_dir, "pytorch_model.bin")
                    torch.save(each_model_to_save.state_dict(), each_output_model_file)
                    logger.info("Save the last model into %s", each_output_model_file)


                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > patience for x in [not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_loss_dec_cnt=%d\n" % (
                            epoch, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

                # #Calculate bleu
                # if 'dev_bleu' in dev_dataset:
                #     eval_examples,eval_data=dev_dataset['dev_bleu']
                # else:
                #     eval_examples = read_examples(args.dev_filename)
                #     eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                #     eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                #     all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                #     eval_data = TensorDataset(all_source_ids)
                #     dev_dataset['dev_bleu'] = eval_examples,eval_data
                #
                # eval_sampler = SequentialSampler(eval_data)
                # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                #
                # model.eval()
                # p=[]
                # for batch in eval_dataloader:
                #     batch = tuple(t.to(device) for t in batch)
                #     source_ids = batch[0]
                #     with torch.no_grad():
                #         preds = model(source_ids)
                #         # convert ids to text
                #         for pred in preds:
                #             t = pred[0].cpu().numpy()
                #             t = list(t)
                #             if 0 in t:
                #                 t = t[:t.index(0)]
                #             text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                #             p.append(text)
                # model.train()
                # predictions = []
                # EM = []
                # # dev.output中是预测结果，dev.gold中是实际内容
                # with open(args.output_dir+"/dev.output",'w') as f, open(args.output_dir+"/dev.gold",'w') as f1:
                #     for ref,gold in zip(p,eval_examples):
                #         predictions.append(ref)
                #         f.write(ref+'\n')
                #         f1.write(gold.target+'\n')
                #         # EM列表中是预测与真实值是否相同的bool值
                #         EM.append(ref.split()==gold.target.split())
                #
                # # 调用bleu._bleu,直接传入两个文件目录进行bleu值的计算
                # dev_bleu = _bleu(args.output_dir+"/dev.gold", args.output_dir+"/dev.output")
                # logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                # logger.info("  %s = %s "%("EM",str(round(np.mean(EM)*100,2))))
                # logger.info("  "+"*"*20)
                # dev_score = dev_bleu+round(np.mean(EM)*100,2)
                # if dev_score>best_score:
                #     logger.info("  Best score:%s",dev_score)
                #     logger.info("  "+"*"*20)
                #     best_score=dev_score
                #     # Save best checkpoint for best bleu
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-best-score')
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     # PyTorch中，如果模型是使用nn.DataParallel进行多GPU训练的，那么模型的参数会被包裹在一个额外的nn.DataParallel模块内，这个额外的模块叫做module
                #     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                #     output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                #     torch.save(model_to_save.state_dict(), output_model_file)
                #     patience =0
                # else:
                #     patience +=1
                #     if patience == -1:
                #         break
    if args.do_test:
        # checkpoint-best-score是一个子目录，在其中保存当前取得最好性能的模型文件
        # checkpoint_prefix = 'checkpoint-best-score/pytorch_model.bin'
        checkpoint_prefix = 'checkpoint-best-ppl/pytorch_model.bin'
        # checkpoint_prefix = 'checkpoint-last/pytorch_model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(output_dir))

        # eval_examples = read_examples(args.test_filename)
        # eval_examples = read_pyclass_examples(args.test_filename)
        eval_examples = read_concode_examples(args.test_filename)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_source_ids)

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        p = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids = batch[0]
            with torch.no_grad():
                preds = model(source_ids)
                # convert ids to text
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)

        args.res_dir = os.path.join(args.output_dir, 'prediction')
        if not os.path.exists(args.res_dir):
            os.makedirs(args.res_dir)

        output_fn = os.path.join(args.res_dir, "test_best_ppl.output")
        gold_fn = os.path.join(args.res_dir, "test_best_ppl.gold")
        src_fn = os.path.join(args.res_dir, "test_best_ppl.src")

        # with open(args.output_dir+"/predictions.txt",'w') as f:
        #     for ref,gold in zip(p,eval_examples):
        #         predictions.append(str(gold.idx)+'\t'+ref)
        #         f.write(ref+'\n')

        # 自己根据训练时的函数构建编写评价过程
        # EM = []
        # with open(args.output_dir+"/predictions.txt",'w') as f, open(args.output_dir+"/predictions.gold",'w') as f1:
        #     for ref,gold in zip(p,eval_examples):
        #         predictions.append(ref)
        #         f.write(ref+'\n')
        #         f1.write(gold.target+'\n')
        #         # EM列表中是预测与真实值是否相同的bool值
        #         EM.append(ref.split()==gold.target.split())
        #
        # # 调用bleu._bleu,直接传入两个文件目录进行bleu值的计算
        # dev_bleu = _bleu(args.output_dir+"/predictions.txt", args.output_dir+"/predictions.gold")
        # logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
        # logger.info("  %s = %s "%("EM",str(round(np.mean(EM)*100,2))))
        # logger.info("  "+"*"*20)
        # predictions_score = dev_bleu+round(np.mean(EM)*100,2)
        # logger.info("  Best score:%s",predictions_score)
        # logger.info("  "+"*"*20)

        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(p, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                # 将类内部的换行符替换为 '\n' 字符
                formatted_pred_nl = pred_nl.replace('\n', '\\n').strip()
                formatted_gold_target = gold.target.replace('\n', '\\n').strip()
                formatted_gold_source = gold.source.replace('\n', '\\n').strip()
                f.write(formatted_pred_nl + '\n')
                f1.write(formatted_gold_target + '\n')
                f2.write(formatted_gold_source + '\n')

        bleu = round(_bleu(gold_fn, output_fn), 2)
        codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, 'python')
        result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
        result['codebleu'] = codebleu * 100
        logger.info("***** Test results *****")
        result_str = "bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (result['bleu'], result['em'], result['codebleu'])
        logger.info(result_str)
        fa.write(result_str)

    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()


