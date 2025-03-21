from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import random
import torch
import time
from tqdm import tqdm
from _utils import *
import torch.nn.functional as F
# import seaborn as sns
# import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# 参数说明
# filename：your_CodeT5_path/CodeT5/data/concode/test.json
# pool：用于多线程处理
# tokenizer：RobertaTokenizer.from_pretrained(model_name_or_path)
# split_tag: 'test'
# only_src=True, is_sample=False
def load_and_cache_gen_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False,data_num=1000):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)

    # 这是训练数据中所有的数据对象
    examples = read_examples(filename, args.data_num, args.task)

    if is_sample:
        examples = random.sample(examples, min(data_num, len(examples)))
    if split_tag == 'train':
        # 此方法对训练数据进行统计，输出分词前后，输入输出数据的长度情况
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 5k data for computing bleu from %s", filename)
        else:
            # Create cache data into
            # saved_models/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/cache_data/test_src_all.pt
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        #  pool.map 函数，该函数通常用于并行地映射函数到输入数据的每个元素上。在这里，convert_examples_to_features 函数被映射到 tuple_examples 中的每个元素上。
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_siamese_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False,data_num=1000):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)

    # 这是训练数据中所有的数据对象
    examples = read_examples(filename, args.data_num, args.task)
    # examples中的每个对象有comment、ast、label字段

    if is_sample:
        examples = random.sample(examples, min(data_num, len(examples)))
    if split_tag == 'train':
        # 此方法对训练数据进行统计，输出分词前后，输入输出数据的长度情况
        calc_siamese_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_siamese_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 5k data for computing bleu from %s", filename)
        else:
            # Create cache data into
            # saved_models/concode/codet5_base_all_lr10_bs16_src320_trg150_pat3_e30/cache_data/test_src_all.pt
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        #  pool.map 函数，该函数通常用于并行地映射函数到输入数据的每个元素上。在这里，convert_examples_to_features 函数被映射到 tuple_examples 中的每个元素上。
        features = pool.map(convert_siamese_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_ast_ids = torch.tensor([f.ast_ids for f in features], dtype=torch.long)
        all_targets = torch.tensor([f.target for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_ast_ids, all_targets)
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_clone_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + '_all' if args.data_num == -1 else '_%d' % args.data_num)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_clone_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_defect_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_multi_gen_data(args, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        examples_data_dict = torch.load(cache_fn)
    else:
        examples_data_dict = {}

        task_list = ['summarize', 'translate', 'refine', 'concode', 'defect']
        for task in task_list:
            if task == 'summarize':
                sub_tasks = ['ruby', 'javascript', 'go', 'python', 'java', 'php']
            elif task == 'translate':
                sub_tasks = ['java-cs', 'cs-java']
            elif task == 'refine':
                sub_tasks = ['small', 'medium']
            else:
                sub_tasks = ['none']
            args.task = task
            for sub_task in sub_tasks:
                args.sub_task = sub_task
                if task == 'summarize':
                    args.max_source_length = 256
                    args.max_target_length = 128
                elif task == 'translate':
                    args.max_source_length = 320
                    args.max_target_length = 256
                elif task == 'refine':
                    if sub_task == 'small':
                        args.max_source_length = 130
                        args.max_target_length = 120
                    else:
                        args.max_source_length = 240
                        args.max_target_length = 240
                elif task == 'concode':
                    args.max_source_length = 320
                    args.max_target_length = 150
                elif task == 'defect':
                    args.max_source_length = 512
                    args.max_target_length = 3  # as do not need to add lang ids

                filename = get_filenames(args.data_dir, args.task, args.sub_task, split_tag)
                examples = read_examples(filename, args.data_num, args.task)
                if is_sample:
                    examples = random.sample(examples, min(5000, len(examples)))
                if split_tag == 'train':
                    calc_stats(examples, tokenizer, is_tokenize=True)
                else:
                    calc_stats(examples)

                tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
                if args.data_num == -1:
                    features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
                else:
                    features = [convert_examples_to_features(x) for x in tuple_examples]
                all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
                if only_src:
                    data = TensorDataset(all_source_ids)
                else:
                    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
                    data = TensorDataset(all_source_ids, all_target_ids)
                examples_data_dict['{}_{}'.format(task, sub_task) if sub_task != 'none' else task] = (examples, data)

        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(examples_data_dict, cache_fn)
            logger.info("Save data into %s", cache_fn)
    return examples_data_dict


# 传入的参数为args.data_dir, args.task, args.sub_task，也即
# data_dir = your_CodeT5_path/CodeT5/data
# task = concode
# subtask = null
def get_filenames(data_root, task, sub_task, split=''):
    if task == 'concode':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.json'.format(data_dir)
        dev_fn = '{}/dev.json'.format(data_dir)
        test_fn = '{}/test.json'.format(data_dir)
        # 想要修改测试文件，在此修改
        # test_fn = '{}/test_gen.json'.format(data_dir)
    elif task == 'summarize':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    elif task == 'refine':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.buggy-fixed.buggy,{}/train.buggy-fixed.fixed'.format(data_dir, data_dir)
        dev_fn = '{}/valid.buggy-fixed.buggy,{}/valid.buggy-fixed.fixed'.format(data_dir, data_dir)
        test_fn = '{}/test.buggy-fixed.buggy,{}/test.buggy-fixed.fixed'.format(data_dir, data_dir)
    elif task == 'translate':
        data_dir = '{}/{}'.format(data_root, task)
        if sub_task == 'cs-java':
            train_fn = '{}/train.java-cs.txt.cs,{}/train.java-cs.txt.java'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.cs,{}/valid.java-cs.txt.java'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.cs,{}/test.java-cs.txt.java'.format(data_dir, data_dir)
        else:
            train_fn = '{}/train.java-cs.txt.java,{}/train.java-cs.txt.cs'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.java,{}/valid.java-cs.txt.cs'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.java,{}/test.java-cs.txt.cs'.format(data_dir, data_dir)
    elif task == 'clone':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.txt'.format(data_dir)
        dev_fn = '{}/valid.txt'.format(data_dir)
        test_fn = '{}/test.txt'.format(data_dir)
    elif task == 'defect':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    # 定义自己的训练任务
    elif task == 'gen_class':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.json'.format(data_dir)
        dev_fn = '{}/dev.json'.format(data_dir)
        test_fn = '{}/test.json'.format(data_dir)
    elif task == 'siamese':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.json'.format(data_dir)
        dev_fn = '{}/dev.json'.format(data_dir)
        test_fn = '{}/test.json'.format(data_dir)
    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task):
    read_example_dict = {
        'summarize': read_summarize_examples,
        'refine': read_refine_examples,
        'translate': read_translate_examples,
        'concode': read_concode_examples,
        'clone': read_clone_examples,
        'defect': read_defect_examples,
        # 定义自己的代码生成任务
        'gen_class': read_gen_class_examples,
        'sia_gen':read_gen_class_examples,
        'siamese': read_siamese_examples
    }
    return read_example_dict[task](filename, data_num)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def calc_siamese_stats(examples, tokenizer=None, is_tokenize=False):
    avg_com_len = []
    avg_ast_len = []
    avg_com_len_tokenize = []
    avg_ast_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_com_len.append(len(ex.comment.split()))
            avg_ast_len.append(len(str(ex.ast).split()))
            avg_com_len_tokenize.append(len(tokenizer.tokenize(ex.comment)))
            avg_ast_len_tokenize.append(len(tokenizer.tokenize(str(ex.ast))))
        else:
            avg_com_len.append(len(ex.comment.split()))
            avg_ast_len.append(len(str(ex.ast).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg_com_len: %d, avg_ast_len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_com_len), np.mean(avg_ast_len), max(avg_com_len), max(avg_ast_len))
        logger.info("[TOKENIZE] avg_com_len: %d, avg_ast_len: %d, max_com_len: %d, max_ast_len: %d",
                    np.mean(avg_com_len_tokenize), np.mean(avg_ast_len_tokenize), max(avg_com_len_tokenize),
                    max(avg_ast_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_com_len), np.mean(avg_ast_len), max(avg_com_len), max(avg_ast_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


def siamese_test_euclidean(pooled_comment_embeddings,pooled_ast_embeddings,target):
    # 初始化变量用于累计欧氏距离
    total_distance_label_1 = 0.0
    total_distance_label_0 = 0.0
    count_label_1 = 0
    count_label_0 = 0

    euclidean_distance = F.pairwise_distance(pooled_comment_embeddings, pooled_ast_embeddings)
    # cosine_similarity = F.cosine_similarity(pooled_comment_embeddings, pooled_ast_embeddings)

    # print(f"Euclidean distances: {euclidean_distance.cpu().numpy()}")
    # 设置阈值和 margin
    # threshold = 0.5
    # margin = 0.9

    # 根据阈值预测标签
    # preds = torch.where(euclidean_distance < threshold, torch.tensor(1, device=args.device),
    #                     torch.where(euclidean_distance > margin, torch.tensor(0, device=args.device),
    #                                 torch.tensor(-1, device=args.device)))

    # preds = torch.where(cosine_similarity > threshold_high, torch.tensor(1, device=args.device),
    #                           torch.where(cosine_similarity < threshold_low, torch.tensor(0, device=args.device),
    #                                       torch.tensor(-1, device=args.device)))

    # 遍历每个样本，累计对应的欧氏距离
    for i in range(len(target)):
        if target[i] == 1:
            total_distance_label_1 += euclidean_distance[i].item()
            count_label_1 += 1
        elif target[i] == 0:
            total_distance_label_0 += euclidean_distance[i].item()
            count_label_0 += 1

        # all_preds.extend(preds.cpu().numpy())
        # all_labels.extend(target.cpu().numpy())

        # 计算平均距离
        avg_distance_label_1 = total_distance_label_1 / count_label_1 if count_label_1 > 0 else 0
        avg_distance_label_0 = total_distance_label_0 / count_label_0 if count_label_0 > 0 else 0

        print(f"Average Euclidean distance when label is 1: {avg_distance_label_1}")  # 0.4240662580603587
        print(f"Average Euclidean distance when label is 0: {avg_distance_label_0}")  # 1.2424643371709736

        # print(f'all_preds:{all_preds}')
        # print(f'all_labels:{all_labels}')
        # 计算准确率
        # correct_predictions = sum(p == l for p, l in zip(all_preds, all_labels))
        # total_predictions = len(all_labels)
        # accuracy = correct_predictions / total_predictions
        # result_str = 'Accuracy: %.2f\n avg_distance_label_1: %.3f\n avg_distance_label_0: %.3f\n' % (
        # 100 * accuracy, avg_distance_label_1, avg_distance_label_0)
        result_str = 'avg_distance_label_1: %.3f\n avg_distance_label_0: %.3f\n' % (
            avg_distance_label_1, avg_distance_label_0)

        return result_str


def siamese_test_cosine_similarity(pooled_comment_embeddings, pooled_ast_embeddings, target):
    # 初始化变量用于累计相似度
    total_similarity_label_1 = 0.0
    total_similarity_label_0 = 0.0
    count_label_1 = 0
    count_label_0 = 0

    cosine_similarity = F.cosine_similarity(pooled_comment_embeddings, pooled_ast_embeddings)

    # 遍历每个样本，累计对应的余弦相似度
    for i in range(len(target)):
        if target[i] == 1:
            total_similarity_label_1 += cosine_similarity[i].item()
            count_label_1 += 1
        elif target[i] == 0:
            total_similarity_label_0 += cosine_similarity[i].item()
            count_label_0 += 1

        # all_preds.extend(preds.cpu().numpy())
        # all_labels.extend(target.cpu().numpy())

        # 计算平均距离
        avg_cosine_similarity_label_1 = total_similarity_label_1 / count_label_1 if count_label_1 > 0 else 0
        avg_cosine_similarity_label_0 = total_similarity_label_0 / count_label_0 if count_label_0 > 0 else 0

        print(f"Average cosine_similarity when label is 1: {avg_cosine_similarity_label_1}")
        print(f"Average cosine_similarity when label is 0: {avg_cosine_similarity_label_0}")

        result_str = 'avg_cosine_similarity_label_1: %.3f\n avg_cosine_similarity_label_0: %.3f\n' % (
            avg_cosine_similarity_label_1, avg_cosine_similarity_label_0)

        return result_str


def filter_encoder_state_dict(state_dict):
    filtered_state_dict = {}
    for k, v in state_dict.items():
        # 过滤掉非编码器部分的参数并移除前缀
        if k.startswith("encoder."):
            filtered_state_dict[k.replace("encoder.", "")] = v
    return filtered_state_dict

def filter_decoder_state_dict(state_dict):
    filtered_state_dict = {}
    for k, v in state_dict.items():
        # 过滤掉非编码器部分的参数并移除前缀
        if k.startswith("decoder."):
            filtered_state_dict[k.replace("decoder.", "")] = v
    return filtered_state_dict


def weighted_edit_distance_loss(gen_seq, target_seq):
    """
    该函数用于使用改进后的编辑距离算法，计算生成ast序列和目标ast序列之间的相似度
    """
    n, m = len(gen_seq), len(target_seq)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # 定义节点类型权重
    weights = {
        "ClassDefinition": 10,  # ClassDefinition层级，权重最高
        "FunctionDefinition": 5,  # FunctionDefinition层级
        "StatementAndClause": 3,  # 语句和分支层级
        "Others": 1  # 其他节点层级
    }

    # 获取节点所属层级的函数
    def get_node_level(node):
        if node.endswith("ClassDefinition"):
            return "ClassDefinition"
        elif node.endswith("FunctionDefinition"):
            return "FunctionDefinition"
        elif node.endswith("Statement") or node.endswith("Clause"):
            return "StatementAndClause"
        else:
            return "Others"

    # 初始化边界条件并记录最大可能的损失
    max_actual_loss = 0

    # 初始化边界条件
    for i in range(1, n + 1):
        weight = weights.get(get_node_level(gen_seq[i - 1]), 1)
        dp[i][0] = dp[i - 1][0] + weight  # 删除操作的加权
        max_actual_loss += weight
    for j in range(1, m + 1):
        weight = weights.get(get_node_level(target_seq[j - 1]), 1)
        dp[0][j] = dp[0][j - 1] + weight  # 插入操作的加权
        max_actual_loss += weight

    # 动态规划计算编辑距离
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gen_seq[i - 1] == target_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 匹配，无损失
            else:
                delete_cost = dp[i - 1][j] + weights.get(get_node_level(gen_seq[i - 1]), 1)
                insert_cost = dp[i][j - 1] + weights.get(get_node_level(target_seq[j - 1]), 1)
                replace_cost = dp[i - 1][j - 1] + max(weights.get(get_node_level(gen_seq[i - 1]), 1),
                                                      weights.get(get_node_level(target_seq[j - 1]), 1))

                dp[i][j] = min(delete_cost, insert_cost, replace_cost)
                max_actual_loss = max(max_actual_loss, dp[i][j])

    # 归一化编辑距离损失
    edit_distance = dp[n][m]
    normalized_loss = edit_distance / max_actual_loss if max_actual_loss else 0  # 避免除0
    return normalized_loss


# def visualize_euclidean_distance_heatmap(distance_matrix):
#     """
#     可视化注释和AST序列之间的欧式距离
#     :param comment_embeddings: 注释的嵌入向量，形状为 (N, D)
#     :param ast_embeddings: AST序列的嵌入向量，形状为 (M, D)
#     """
#     # 计算欧式距离矩阵
#     # distance_matrix = calculate_euclidean_distance(comment_embeddings, ast_embeddings)
#
#     # 绘制热图
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(distance_matrix, cmap="YlGnBu", xticklabels=False, yticklabels=False)
#     plt.title("Euclidean Distance Heatmap Between Comment and AST Embeddings")
#     plt.xlabel("AST Embeddings")
#     plt.ylabel("Comment Embeddings")
#     plt.show()