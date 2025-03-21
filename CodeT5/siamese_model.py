import torch
import torch.nn as nn
import torch.nn.functional as F



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
# 损失函数来自Yann LeCun 的 Dimensionality Reduction by Learning an Invariant Mapping

# 使用余弦相似度计算
# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, output1, output2, label):
#         # 计算余弦相似度
#         cosine_similarity = F.cosine_similarity(output1, output2)
#         # 相似样本的损失
#         pos_loss = label * torch.pow(1 - cosine_similarity, 2)
#         # 不相似样本的损失
#         neg_loss = (1 - label) * torch.pow(torch.abs(cosine_similarity), 2)
#         # 计算总损失
#         loss = 0.5 * torch.mean(pos_loss + neg_loss)
#         return loss


class SiameseEncoder(nn.Module):
    def __init__(self,encoder):
        super(SiameseEncoder, self).__init__()
        self.encoder = encoder

    def forward(self,input_ids,attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # 获取编码器的最后隐藏状态
        return last_hidden_state


