import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from SPTCode.ast_parser import  generate_single_enhanced_ast
import json

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

def read_concode_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
    return examples


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # 计算欧式距离
        euclidean_distance = F.pairwise_distance(output1, output2)
        # 相似样本的损失
        pos_loss = label * torch.pow(euclidean_distance, 2)
        # 不相似样本的损失
        neg_loss = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        # 计算总损失
        loss = 0.5 * torch.mean(pos_loss + neg_loss)
        return loss


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)

def filter_encoder_state_dict(state_dict):
    filtered_state_dict = {}
    for k, v in state_dict.items():
        # 过滤掉非编码器部分的参数并移除前缀
        if k.startswith("siamese."):
            filtered_state_dict[k.replace("siamese.", "")] = v
    return filtered_state_dict

def filter_Roberta_encoder_state_dict(state_dict):
    filtered_state_dict = {}
    for k, v in state_dict.items():
        # 只保留encoder相关的层，排除embeddings和pooler
        if k.startswith("encoder.") and not any(layer in k for layer in ["embeddings", "pooler"]):
            filtered_state_dict[k.replace("encoder.", "")] = v
    return filtered_state_dict


def calculate_ast_loss(outputs,target_ids,tokenizer):
    predicted_ids = torch.argmax(outputs, dim=-1)  # 获取每个时间步的最高概率标记

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