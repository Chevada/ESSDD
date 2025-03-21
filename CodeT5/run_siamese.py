import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_siamese_data,siamese_test_euclidean,siamese_test_cosine_similarity
from configs import add_args, set_seed, set_dist
import pickle

from siamese_model import SiameseEncoder,ContrastiveLoss


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer,criterion):
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
        source_ids, ast_ids, target = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        ast_mask = ast_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            comment_embeddings = model(source_ids, source_mask)
            ast_embeddings = model(ast_ids, ast_mask)

            pooled_comment_embeddings = torch.mean(comment_embeddings, dim=1)
            pooled_ast_embeddings = torch.mean(ast_embeddings, dim=1)

            loss = criterion(pooled_comment_embeddings, pooled_ast_embeddings, target)

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        eval_loss += loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, t5_model, tokenizer = build_or_load_gen_model(args)
    t5_encoder = t5_model.get_encoder()
    semodel = SiameseEncoder(t5_encoder)
    semodel.to(args.device)

    # 冻结原有编码器的参数
    for param in semodel.encoder.parameters():
        param.requires_grad = False

    # 解冻编码器的后几层参数
    # num_layers_to_unfreeze = 6  # 你想要解冻的层数，可以调整
    # encoder_layers = semodel.encoder.block

    # 解冻最顶部的2层,采用渐进解冻的方式进行训练，在训练初期，先冻结编码器的大部分层，只更新少量的顶层参数。随着训练的进行，逐步解冻更多层，直到训练的最后才解冻底层参数。
    # 开始只保留两层参数，后续每训练4个轮次，就解冻两层的参数
    num_layers_to_unfreeze = 2
    encoder_layers = semodel.encoder.block

    for layer in encoder_layers[-num_layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True

    if args.n_gpu > 1:
        # for DataParallel
        semodel = torch.nn.DataParallel(semodel)
    pool = multiprocessing.Pool(args.cpu_cont)
    # args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)

    data_dir = '{}/{}'.format(args.data_dir, 'siamese')
    args.train_filename = '{}/train.json'.format(data_dir)
    args.dev_filename = '{}/dev.json'.format(data_dir)
    args.test_filename = '{}/test.json'.format(data_dir)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_siamese_data(args, args.train_filename, pool, tokenizer, 'train',is_sample=True,data_num=40000)
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        optimizer = AdamW(filter(lambda p: p.requires_grad, semodel.parameters()),lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        criterion = ContrastiveLoss(margin=1.0)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_lost,best_ppl= 0,100,1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6
        loss_list = []

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            semodel.train()

            cur_loss,best_loss = 0,100

            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, ast_ids,target = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                ast_mask = ast_ids.ne(tokenizer.pad_token_id)

                comment_embeddings = semodel(source_ids,source_mask)
                ast_embeddings = semodel(ast_ids,ast_mask)

                pooled_comment_embeddings = torch.mean(comment_embeddings, dim=1)
                pooled_ast_embeddings = torch.mean(ast_embeddings, dim=1)

                loss = criterion(pooled_comment_embeddings,pooled_ast_embeddings,target)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                # 用于记录训练过程中已处理的样本数量
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
                    cur_loss = train_loss
                # 记录当前轮次的最终损失
            epoch_train_loss = round(tr_loss / nb_tr_steps, 4)
            loss_list.append(epoch_train_loss)

            # 防止训练过拟合
            # Eval model with dev data
            if args.do_eval:
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_siamese_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, semodel, tokenizer,criterion)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)


                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    if cur_loss < best_lost:
                        best_lost = cur_loss
                        model_to_save = semodel.module if hasattr(semodel, 'module') else semodel
                        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the last model into %s,the min loss is %f", output_model_file,best_lost)
                    else:
                        logger.info("%d epoch loss does not decrease", cur_epoch)
                        break
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
                        model_to_save = semodel.module if hasattr(semodel, 'module') else semodel
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

            # 每 4 个 epoch 解冻 2 层
            if (cur_epoch + 1) % 4 == 0 and num_layers_to_unfreeze < 8:
                num_layers_to_unfreeze += 2
                encoder_layers = semodel.module.encoder.block if isinstance(semodel,
                                                                            torch.nn.DataParallel) else semodel.encoder.block
                num_layers_to_unfreeze = min(num_layers_to_unfreeze, len(encoder_layers))

                for layer in encoder_layers[-num_layers_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        # 单独在测试数据集上进行验证时，重新制定下输出文件夹

        # file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format('best-ppl'))
        file = os.path.join('/AIsoftwaremfq2023/code/cl_code/CodeT5/sh/saved_models/siamese/codet5_base_all_useeuc_lr10_bs40_src64_trg150_pat2_e30_20241020_071605', 'checkpoint-{}/pytorch_model.bin'.format('best-ppl'))
        # model.load_state_dict(torch.load(file))
        semodel.module.load_state_dict(torch.load(file))
        test_examples, test_data = load_and_cache_siamese_data(args, args.test_filename, pool, tokenizer, 'test')

        logger.info("  ***** Running bleu evaluation on test data*****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_sampler = SequentialSampler(test_data)
        if args.data_num == -1:
            eval_dataloader = DataLoader(test_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                         num_workers=4, pin_memory=True)
        else:
            eval_dataloader = DataLoader(test_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        semodel.eval()

        # total_similarity_label_1 = 0.0
        # total_similarity_label_0 = 0.0
        # count_label_1 = 0
        # count_label_0 = 0

        total_distance_label_1 = 0.0
        total_distance_label_0 = 0.0
        count_label_1 = 0
        count_label_0 = 0

        # 用于保存目标为1和0的列表
        positive_euclidean_distance_samples = []
        negative_euclidean_distance_samples = []

        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for test set"):
            with torch.no_grad():
                source_ids, ast_ids, target = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                ast_mask = ast_ids.ne(tokenizer.pad_token_id)

                comment_embeddings = semodel(source_ids, source_mask)
                ast_embeddings = semodel(ast_ids, ast_mask)

                pooled_comment_embeddings = torch.mean(comment_embeddings, dim=1)
                pooled_ast_embeddings = torch.mean(ast_embeddings, dim=1)

                # 使用欧式距离
                euclidean_distance = F.pairwise_distance(pooled_comment_embeddings, pooled_ast_embeddings)

                # 按目标值分组
                for i, tgt in enumerate(target):
                    if tgt == 1 and len(positive_euclidean_distance_samples) < 40:
                        positive_euclidean_distance_samples.append(euclidean_distance[i].cpu().numpy())
                    elif tgt == 0 and len(negative_euclidean_distance_samples) < 40:
                        negative_euclidean_distance_samples.append(euclidean_distance[i].cpu().numpy())
            # 如果两个列表都已满，停止保存
            if len(positive_euclidean_distance_samples) >= 40 and len(negative_euclidean_distance_samples) >= 40:
                break

                # for i in range(len(target)):
                #     if target[i] == 1:
                #         total_distance_label_1 += euclidean_distance[i].item()
                #         count_label_1 += 1
                #     elif target[i] == 0:
                #         total_distance_label_0 += euclidean_distance[i].item()
                #         count_label_0 += 1


        # avg_distance_label_1 = total_distance_label_1 / count_label_1 if count_label_1 > 0 else 0
        # avg_distance_label_0 = total_distance_label_0 / count_label_0 if count_label_0 > 0 else 0
        #
        # print(f"Average Euclidean distance when label is 1: {avg_distance_label_1}")
        # print(f"Average Euclidean distance when label is 0: {avg_distance_label_0}")
        #
        # result_str = 'avg_distance_label_1: %.3f\n avg_distance_label_0: %.3f\n' % (
        #     avg_distance_label_1, avg_distance_label_0)

        # print(f"Average cosine_similarity when label is 1: {avg_cosine_similarity_label_1}")
        # print(f"Average cosine_similarity when label is 0: {avg_cosine_similarity_label_0}")

        # result_str = 'avg_cosine_similarity_label_1: %.3f\n avg_cosine_similarity_label_0: %.3f\n' % (
        #     avg_cosine_similarity_label_1, avg_cosine_similarity_label_0)

        # logger.info(result_str)
        # fa.write(result_str)
        # if args.res_fn:
        #     with open(args.res_fn, 'a+') as f:
        #         f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
        #         f.write(result_str)

        # 将正负样本保存为 NumPy 数组
        positive_samples = np.array(positive_euclidean_distance_samples)
        negative_samples = np.array(negative_euclidean_distance_samples)

        pos_dir = os.path.join(args.output_dir, 'positive_samples.npy')
        neg_dir = os.path.join(args.output_dir, 'negative_samples.npy')

        # 保存为文件（可以选择保存为 npy 文件或其他格式）
        np.save(pos_dir, positive_samples)
        np.save(neg_dir, negative_samples)
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()

if __name__ == '__main__':
    main()