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
import math
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
from utils import filter_encoder_state_dict,get_elapse_time,filter_Roberta_encoder_state_dict

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
    # params_path = 'saved_models/siamese'
    # params_file = os.path.join(params_path, 'checkpoint-{}/pytorch_model.bin'.format('best-ppl'))
    # pretrained_encoder_state_dict = filter_encoder_state_dict(torch.load(params_file))
    # model.module.encoder.load_state_dict(pretrained_encoder_state_dict)
    #
    # logger.info(f"Successfully imported parameters from siamese network {params_file}")

    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        # 加载数据集
        pooled_logits_list = []
        ast_loss_list = []
        data_dir = '/AIsoftwaremfq2023/code/cl_code/UniXcoder/code-generation/dataset/ast_loss_data/concode'
        files = os.listdir(data_dir)
        data_files = [f for f in files if f.endswith('.pt')]
        for file in data_files:
            # Load the saved data from each file
            file_path = os.path.join(data_dir, file)
            data = torch.load(file_path)

            # Append the pooled_logits and ast_loss
            pooled_logits_list.append(data['pooled_logits'])
            ast_loss_list.append(data['ast_loss'])
        pooled_logits = torch.cat(pooled_logits_list, dim=0)
        ast_loss = torch.cat(ast_loss_list, dim=0)

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

        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            fcn_loss_total = 0  # 初始化累计损失
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
                    epoch, round(fcn_loss_total / (step + 1), 3) if step > 0 else 0
                ))

                output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if fcn_loss_total / (step + 1) < 0.01 or epoch > 10:
                    logger.info("Stopping training at epoch {} with FCN loss {}".format(epoch, fcn_loss))
                    # 保存模型参数
                    torch.save(astLossModel.state_dict(),
                               output_model_file)
                    break  # 结束当前轮次的训练

            if args.do_eval:
                pass
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        logger.info("Finish training and take %s", get_elapse_time(t0))
    if args.do_test:
        # checkpoint-best-score是一个子目录，在其中保存当前取得最好性能的模型文件
        # checkpoint_prefix = 'checkpoint-best-score/pytorch_model.bin'
        checkpoint_prefix = 'checkpoint-best-ppl/pytorch_model.bin'
        # checkpoint_prefix = 'checkpoint-last/pytorch_model.bin'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(output_dir))

        # eval_examples = read_examples(args.test_filename)
        eval_examples = read_pyclass_examples(args.test_filename)
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


