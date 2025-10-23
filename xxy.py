import math
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

    def forward(self, self_vectors, neighbor_vector, batch_size, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
        output = torch.cat([self_vectors, neighbor_vector], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        return output


class CombineGraph(Module):
    def __init__(self, opt, num_node, num, batch_size, hidden_size, dropout_global, n_iter, n_sample):
        super(CombineGraph, self).__init__()
        self.batch_size = batch_size
        self.num_node = num_node
        self.dim = hidden_size
        self.dropout_global = dropout_global
        self.hop = n_iter
        self.sample_num = n_sample
        self.num = trans_to_cuda(torch.Tensor(num)).float()

        # Aggregator

        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def forward(self, inputs, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]

        # global
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)

        # mean 
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)

        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop + 1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)

        return h_global

