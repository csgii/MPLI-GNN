import datetime
import math
import numpy as np
import torch
from entmax import entmax_bisect
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class FindNeighbors(Module):
    def __init__(self, hidden_size):
        super(FindNeighbors, self).__init__()
        self.hidden_size = hidden_size
        self.neighbor_n = 3 # Diginetica:3; Tmall: 7; Nowplaying: 4
        self.dropout40 = nn.Dropout(0.40)

    def compute_sim(self, sess_emb):
        fenzi = torch.matmul(sess_emb, sess_emb.permute(1, 0))
        fenmu_l = torch.sum(sess_emb * sess_emb + 0.000001, 1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))
        cos_sim = fenzi / fenmu
        cos_sim = nn.Softmax(dim=-1)(cos_sim)
        return cos_sim

    def forward(self, sess_emb):
        k_v = self.neighbor_n
        cos_sim = self.compute_sim(sess_emb)
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = sess_emb[topk_indice]

        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.hidden_size)

        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        neighbor_sess = self.dropout40(neighbor_sess)  # [b,d]
        return neighbor_sess


class LastAttenion(Module):
    def __init__(self, hidden_size, heads, dot, l_p, last_k=3, use_attn_conv=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.last_k = last_k
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_five = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = 0.1
        self.dot = dot
        self.l_p = l_p
        self.use_attn_conv = use_attn_conv
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def forward(self, ht1, hidden, mask):
        q0 = self.linear_zero(ht1).view(-1, ht1.size(1), self.hidden_size // self.heads)
        q1 = self.linear_one(hidden).view(-1, hidden.size(1), self.hidden_size // self.heads)
        q2 = self.linear_two(hidden).view(-1, hidden.size(1), self.hidden_size // self.heads)
        #assert not torch.isnan(q0).any()
        #assert not torch.isnan(q1).any()
        alpha = torch.sigmoid(torch.matmul(q0, q1.permute(0, 2, 1)))
        #assert not torch.isnan(alpha).any()
        alpha = alpha.view(-1, q0.size(1) * self.heads, hidden.size(1)).permute(0, 2, 1)
        alpha = torch.softmax(2 * alpha, dim=1)
        #assert not torch.isnan(alpha).any()
        if self.use_attn_conv == "True":
            m = torch.nn.LPPool1d(self.l_p, self.last_k, stride=self.last_k)
            alpha = m(alpha)
            alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))
            alpha = torch.softmax(2 * alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        a = torch.sum(
            (alpha.unsqueeze(-1) * q2.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(),1)
        

        #a = (alpha.unsqueeze(-1) * q2.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float()
        
        return a



class RelationGAT(Module):
    def __init__(self, batch_size, hidden_size=100):
        super(RelationGAT, self).__init__()
        self.batch_size = batch_size
        self.dim = hidden_size
        self.w_f = nn.Linear(hidden_size, hidden_size)
        self.alpha_w = nn.Linear(self.dim, 1)
        self.atten_w0 = nn.Parameter(torch.Tensor(1, self.dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_bias = nn.Parameter(torch.Tensor(self.dim))

    def get_alpha(self, x=None):
        # x[b,1,d]
        alpha_global = torch.sigmoid(self.alpha_w(x)) + 1  #[b,1,1]
        alpha_global = self.add_value(alpha_global)
        return alpha_global #[b,1,1]


    def add_value(self, value):
        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value


    def tglobal_attention(self, target, k, v, alpha_ent=1):
        alpha = torch.matmul(torch.relu(k.matmul(self.atten_w1) + target.matmul(self.atten_w2) + self.atten_bias),self.atten_w0.t())
        alpha = entmax_bisect(alpha, alpha_ent, dim=1)
        c = torch.matmul(alpha.transpose(1, 2), v)
        return c

    def forward(self, item_embedding, items, A, D, target_embedding):
        seq_h = []
        for i in torch.arange(items.shape[0]):
            seq_h.append(torch.index_select(item_embedding, 0, items[i]))  # [b,s,d]
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        len = seq_h1.shape[1]
        relation_emb_gcn = torch.sum(seq_h1, 1) #[b,d]
        DA = torch.mm(D, A).float() #[b,b]
        relation_emb_gcn = torch.mm(DA, relation_emb_gcn) #[b,d]
        relation_emb_gcn = relation_emb_gcn.unsqueeze(1).expand(relation_emb_gcn.shape[0], len, relation_emb_gcn.shape[1]) #[b,s,d]

        target_emb = self.w_f(target_embedding)
        alpha_line = self.get_alpha(x=target_emb)
        q = target_emb #[b,1,d]
        k = relation_emb_gcn #[b,1,d]
        v = relation_emb_gcn #[b,1,d]

        line_c = self.tglobal_attention(q, k, v, alpha_ent=alpha_line) #[b,1,d]
        c = torch.selu(line_c).squeeze()
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))

        return l_c


class Global_GNN(Module):
    def __init__(self, fuse_A, hidden_size, step=1):
        super(Global_GNN, self).__init__()
        self.fuse_A = fuse_A
        self.step = step
        self.hidden_size = hidden_size
        self.w_h = Parameter(torch.Tensor(self.hidden_size*3, self.hidden_size))
        self.w_hf = Parameter(torch.Tensor(self.hidden_size*2, self.hidden_size))

    def GateCell(self, A, hidden):
        hidden_w = F.linear(hidden, self.w_h)

        hidden_w_0, hidden_w_1, hidden_w_2 = hidden_w.chunk(3, -1)
        hidden_fuse_0, hidden_fuse_1 = F.linear(torch.matmul(A, hidden_w_0), self.w_hf).chunk(2, -1)

        gate = torch.relu(hidden_fuse_0 + hidden_w_1)
        return hidden_w_2 + gate * hidden_fuse_1

    def Fuse_with_correlation(self, A_Global, hidden):
        correlation_A = torch.matmul(hidden, hidden.transpose(1, 0))
        correlation_A_std = torch.norm(correlation_A, p=2, dim=1, keepdim=True)
        correlation_A = correlation_A/correlation_A_std
        return A_Global + correlation_A

    def forward(self, A_Global, hidden):
        seqs = []
        if self.fuse_A:
            A_Global = self.Fuse_with_correlation(A_Global, hidden)
        for i in range(self.step):
            hidden = self.GateCell(A_Global, hidden)
            seqs.append(hidden)
        return hidden, torch.mean(torch.stack(seqs, dim=1), dim=1)


class Global_ATT(Module):
    def __init__(self, hidden_size, num_heads, dp_att, dp_ffn):
        super(Global_ATT, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_att = dp_att
        self.q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(self.dropout_att)
        self.ffn = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dropout_ffn = dp_ffn
        self.ffn_dropout = nn.Dropout(self.dropout_ffn)

    def transpose_qkv(self, hidden):
        hidden = hidden.reshape(hidden.shape[0], self.num_heads, -1)
        hidden = hidden.permute(1, 0, 2)
        return hidden

    def transpose_output(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        inputs = inputs.contiguous().view(inputs.size()[0], 1, -1).squeeze(1)
        return inputs

    def forward(self, inputs):
        query_h = self.transpose_qkv(self.q(inputs))
        key_h = self.transpose_qkv(self.k(inputs))
        value_h = self.transpose_qkv(self.v(inputs))
        softmax_score = torch.tanh(torch.matmul(query_h, key_h.transpose(dim0=1, dim1=-1)))
        att_hidden = self.transpose_output(torch.matmul(self.attention_dropout(softmax_score), value_h))
        att_hidden = self.ffn_dropout(torch.relu(self.ffn(att_hidden)))
        return inputs, att_hidden


class Global_ATT_blocks(Module):
    def __init__(self, block_nums_global, hidden_size, global_att_num_heads, dropout_global_att, dropout_global_ffn):
        super(Global_ATT_blocks, self).__init__()
        self.block_nums = block_nums_global
        self.att_layer = Global_ATT(hidden_size, global_att_num_heads, dropout_global_att, dropout_global_ffn)
        # self.multi_block_att = [self.att_layer for _ in range(self.block_nums)]
        self.multi_block_att = [Global_ATT(hidden_size, global_att_num_heads, dropout_global_att, dropout_global_ffn) for _ in range(self.block_nums)]
        for i, global_attention in enumerate(self.multi_block_att):
            self.add_module('global_attention_{}'.format(i), global_attention)

    def forward(self, inputs):
        for global_att_temp in self.multi_block_att:
            inputs, att_hidden = global_att_temp(inputs)
            inputs = inputs + att_hidden
        return inputs


class DGNN(Module):
    def __init__(self, opt, n_node):
        super(DGNN, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize

        self.global_att_block_nums = opt.global_att_block_nums
        self.global_att_num_heads = opt.global_att_head_nums
        self.dropout_global_att = opt.dropout_global_att
        self.dropout_global_ffn = opt.dropout_global_ffn
        self.fuse_A = opt.fuse_A

        self.embedding = nn.Embedding(self.n_node, self.hidden_size)


        self.global_gnn = Global_GNN(self.fuse_A, self.hidden_size, step=opt.step_global)
        self.global_att_blocks = Global_ATT_blocks(self.global_att_block_nums, self.hidden_size, self.global_att_num_heads, self.dropout_global_att,
                                                   self.dropout_global_ffn)

        self.global_fuse_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.convariance = True
        self.normalization_session = nn.BatchNorm1d(opt.len_max)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.embedding_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.nonhybrid = opt.nonhybrid
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=opt.lr, momentum=opt.mt)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

        self.activate = F.relu
        dim = self.hidden_size
        self.attention_mlp = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.2)
        self.num_attention_heads = opt.num_attention_heads
        self.attention_head_size = int(dim / self.num_attention_heads)
        self.self_atten_w1 = nn.Linear(dim, dim)
        self.self_atten_w2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)
        self.alpha_w = nn.Linear(dim, 1)
        self.multi_alpha_w = nn.Linear(self.attention_head_size, 1)
        self.atten_w0 = nn.Parameter(torch.Tensor(1, dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_bias = nn.Parameter(torch.Tensor(dim))
        self.w_f = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.dim = dim
        self.RelationGraph = RelationGAT(self.batch_size, self.hidden_size)
        self.last_k = opt.last_k
        self.l_p = opt.l_p
        self.use_attn_conv = opt.use_attn_conv
        self.scale = opt.scale
        self.heads = opt.heads
        self.dot = opt.dot
        self.linear_q = nn.ModuleList()
        for i in range(self.last_k):
            self.linear_q.append(nn.Linear((i + 1) * self.dim, self.dim))
        self.mattn = LastAttenion(self.dim, self.heads, self.dot, self.l_p, last_k=self.last_k, use_attn_conv=self.use_attn_conv)
        self.is_dropout = True
        self.FindNeighbor = FindNeighbors(self.hidden_size)
        self.zeros = torch.Tensor(self.batch_size, 1, self.hidden_size).fill_(0).float().cuda()  # [b,i1,d]

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def highway_network(self, hidden, h_0):
        g = torch.sigmoid(self.w_f(torch.cat((h_0, hidden), 1)))  # B,L,d
        h_f = g * h_0 + (1 - g) * hidden
        return h_f


    def add_value(self, value):

        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def Multi_Self_attention(self, q, k, v, sess_len):
        is_dropout = True
        if is_dropout:
            q_ = self.dropout(self.activate(self.attention_mlp(q)))  # [b,s+1,d]
        else:
            q_ = self.activate(self.attention_mlp(q))

        query_layer = self.transpose_for_scores(q_)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        alpha_ent = self.get_alpha2(query_layer[:, :, -1, :], seq_len=sess_len)

        attention_probs = entmax_bisect(attention_scores, alpha_ent, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim,)
        att_v = context_layer.view(*new_context_layer_shape)

        if is_dropout:
            att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(att_v)))) + att_v
        else:
            att_v = self.self_atten_w2(self.activate(self.self_atten_w1(att_v))) + att_v

        att_v = self.LN(att_v)
        c = att_v[:, -1, :].unsqueeze(1)  # [b,d]->[b,1,d]
        x_n = att_v[:, :-1, :]  # [b,s,d]
        return c, x_n

    def get_alpha(self, x=None, seq_len=39, number=None):  # x[b,1,d], seq = len为每个会话序列中最后一个元素
        if number == 0:
            alpha_ent = torch.sigmoid(self.alpha_w(x)) + 1  # [b,1,1]
            alpha_ent = self.add_value(alpha_ent).unsqueeze(1)  # [b,1,1]
            alpha_ent = alpha_ent.expand(-1, seq_len, -1)  # [b,s+1,1]
            return alpha_ent
        if number == 1:  # x[b,1,d]
            alpha_global = torch.sigmoid(self.alpha_w(x)) + 1  # [b,1,1]
            alpha_global = self.add_value(alpha_global)
            return alpha_global


    def get_alpha2(self, x=None, seq_len=39): #x [b,n,d/n]
        alpha_ent = torch.sigmoid(self.multi_alpha_w(x)) + 1  # [b,n,1]
        alpha_ent = self.add_value(alpha_ent).unsqueeze(2)  # [b,n,1,1]
        alpha_ent = alpha_ent.expand(-1, -1, seq_len, -1)  # [b,n,s,1]
        return alpha_ent

    def global_attention(self, target, k, v, mask=None, alpha_ent=1):
        alpha = torch.matmul(torch.relu(k.matmul(self.atten_w1) + target.matmul(self.atten_w2) + self.atten_bias), self.atten_w0.t())
        if mask is not None: #[b,s]
            mask = mask.unsqueeze(-1)
            alpha = alpha.masked_fill(mask == 0, -np.inf)
        alpha = entmax_bisect(alpha, alpha_ent, dim=1)
        c = torch.matmul(alpha.transpose(1, 2), v)
        return c


    # [b,d], [b,d]
    def decoder(self, global_s, target_s):
        if self.is_dropout:
            c = self.dropout(torch.selu(self.w_f(torch.cat((global_s, target_s), 2))))
        else:
            c = torch.selu(self.w_f(torch.cat((global_s, target_s), 2)))  # [b,1,4d]

        c = c.squeeze() #[b,d]
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        return l_c


    def compute_scores(self, hidden, mask, session_target, A_hat, D_hat, inputs):
        sess_len = session_target.shape[1]
        target_emb, x_n = self.Multi_Self_attention(session_target, session_target, session_target, sess_len)
        #relation_emb = self.RelationGraph(self.embedding.weight, inputs, A_hat, D_hat, target_emb)
        relation_emb = target_emb.squeeze(1)



        if self.convariance:
            hidden_std = self.normalization_session(hidden)
            convariance = torch.matmul(hidden_std, hidden_std.transpose(2, 1))
            hidden_conv = torch.matmul(convariance, hidden)
            hidden = hidden + hidden_conv
        
        '''
        hts = []
        lengths = torch.sum(mask, dim=1)
        for i in range(self.last_k):
            hts.append(self.linear_q[i](torch.cat(
                [hidden[torch.arange(mask.size(0)).long(), torch.clamp(lengths - (j + 1), -1, 1000)] for j in
                 range(i + 1)], dim=-1)).unsqueeze(1))
        ht0 = hidden[torch.arange(mask.size(0)).long(), torch.sum(mask, 1) - 1]
        hts = torch.cat(hts, dim=1)
        hts = hts.div(torch.norm(hts, p=2, dim=1, keepdim=True) + 1e-12)
        hidden = hidden[:, :mask.size(1)]
        ais = self.mattn(hts, hidden, mask)
        '''

        #ht0 = self.linear_one(ht0).view(ht0.shape[0], 1, ht0.shape[1])

        #sess_global = torch.sigmoid(ais + ht0)


         
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)
        
        #sess_global = torch.sigmoid(q1 + q2)  # [b,s,d]
        
        #alpha = self.linear_three(torch.sigmoid(ais + ht0))
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        #a = a + relation_emb
        #a = self.linear_transform(torch.cat((ais.squeeze(), ht0), 1))
        a = self.highway_network(a, relation_emb)


        '''
        # Sparse Global Attention
        alpha_global = self.get_alpha(x=target_emb, number=1)  # [b,1,2d]
        q = target_emb
        k = x_n # [b,s,2d]
        v = sess_global  # [b,s,2d]
        global_c = self.global_attention(q, k, v, mask=mask, alpha_ent=alpha_global)
        sess_final = self.decoder(global_c, target_emb)
        # SIC
        neighbor_sess = self.FindNeighbor(sess_final + relation_emb)
        '''
        #sess_final = sess_global + relation_emb

        #alpha = self.linear_three(torch.sigmoid(q1 + q2))
        #a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        #if not self.nonhybrid:
            #a = self.linear_transform(torch.cat([a, ht], 1))
        # b = self.embedding_linear(self.embedding.weight)[1:]
        b = self.embedding.weight[1:]
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, A_global, inputs_global, inputs_global_index, alias_inputs):
        hidden_global = self.embedding(inputs_global)
        hidden_last_global, hidden_fuse_global = self.global_gnn(A_global, hidden_global)
        hidden_att_global = self.global_att_blocks(hidden_global)
        hidden_last_global = self.global_fuse_linear(torch.cat([hidden_last_global, hidden_att_global], dim=-1))
        fuse_global = hidden_last_global[inputs_global_index]


        hidden = fuse_global


        #get = lambda i: hidden[i][alias_inputs[i]]
        #hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden_gnn = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])  # [b,s,2d]
        session_target = torch.cat([seq_hidden_gnn, self.zeros], 1)


        return seq_hidden_gnn, session_target


def forward(model, i, data):
    alias_inputs, mask, targets, A_global, inputs, global_inputs_index, items = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    A_hat, D_hat = data.get_overlap(items)
    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))
    mask = trans_to_cuda(torch.Tensor(mask).long())
    A_global = trans_to_cuda(torch.Tensor(A_global).float())
    inputs = trans_to_cuda(torch.Tensor(inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    global_inputs_index = trans_to_cuda(torch.Tensor(global_inputs_index).long())
    hidden, sess_target = model(A_global, inputs, global_inputs_index, alias_inputs)
    #get = lambda i: hidden[i][alias_inputs[i]]
    #hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    scores = model.compute_scores(hidden, mask, sess_target, A_hat, D_hat, items)
    return targets, scores


def train_test(model, train_data, test_data, seed_random, logger):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    logger.info('start training:{}'.format(datetime.datetime.now()))
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size, seed_random)

    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()

        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
            logger.info('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    logger.info('\tLoss:\t%.3f' % total_loss)
    print('start predicting: ', datetime.datetime.now())
    logger.info('start predicting: {}'.format(datetime.datetime.now()))
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size, seed_random)

    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr, model
