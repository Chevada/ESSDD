# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 文本生成任务中常用的解码策略Beam Search（集束搜索）
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        #register_buffer的作用是将注册到模型的 buffers() 属性中，并命名为bias,这代表bias对应的是一个持久态，不会有梯度传播给它，但是能被模型的state_dict记录下来。可以理解为模型的常数。
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #lm_head 则被用于在预训练之后，对这些表示进行微调，使得模型具备生成文本的能力
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 此lm_head层为线性层，其权重被设置为和编码器中的词嵌入层权重相同
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id       
        
    def forward(self, source_ids, target_ids=None):   
        if target_ids is None:
            # 若是测试，直接生成目标代码
            return self.generate(source_ids)
        
        # ne：not equal比较运算符
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)  
        ids = torch.cat((source_ids,target_ids),-1)
        mask = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
        mask = mask & ids[:,None,:].ne(1)

        out = self.decoder(target_ids,attention_mask=mask,past_key_values=encoder_output.past_key_values).last_hidden_state
        lm_logits = self.lm_head(out)
        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        outputs = loss,loss*active_loss.sum(),active_loss.sum(),lm_logits
        return outputs

    def get_hidden_state(self, source_ids, target_ids=None):
        if target_ids is None:
            # 若是测试，直接生成目标代码
            return self.generate(source_ids)

        # ne：not equal比较运算符
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)
        ids = torch.cat((source_ids,target_ids),-1)
        mask = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
        mask = mask & ids[:,None,:].ne(1)

        decoder_output = self.decoder(target_ids,attention_mask=mask,past_key_values=encoder_output.past_key_values).last_hidden_state
        lm_logits = self.lm_head(decoder_output)
        return lm_logits
    
    def generate(self, source_ids):
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)        
        preds = []       
        zero = torch.cuda.LongTensor(1).fill_(0)   
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                     for y in encoder_output.past_key_values]
            beam = Beam(self.beam_size,self.sos_id,self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
            for _ in range(self.max_length): 
                if beam.done():
                    break

                ids = torch.cat((context_ids,input_ids),-1)
                mask = self.bias[:,context_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
                mask = mask & ids[:,None,:].ne(1)
                out = self.decoder(input_ids,attention_mask=mask,past_key_values=context).last_hidden_state
                hidden_states = out[:,-1,:]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids,beam.getCurrentState()),-1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))

        preds = torch.cat(preds,0)    

        return preds   
        
        
# 束搜索器，通过维护多个候选序列，提高生成序列的质量和多样性
class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence



# 使用编码器构建孪生网络
class SiameseEncoder(nn.Module):
    def __init__(self,encoder):
        super(SiameseEncoder, self).__init__()
        self.siamese = encoder

    def forward(self,source_ids):
        mask = source_ids.ne(1)[:, None, :] * source_ids.ne(1)[:, :, None]
        siamese_output = self.siamese(source_ids, attention_mask=mask, use_cache=True)
        return siamese_output

class ASTLossPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ASTLossPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch Normalization
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出为一个标量，即预测的AST损失

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        # 使用He初始化
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.01)  # 输出层可以使用较小的标准差

        # 偏置初始化为零
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))  # 加入批标准化
        x = torch.relu(self.bn2(self.fc2(x)))
        # x = torch.sigmoid(self.fc3(x))  # 使用sigmoid来约束输出范围
        x = self.fc3(x)  # 使用sigmoid来约束输出范围
        return x