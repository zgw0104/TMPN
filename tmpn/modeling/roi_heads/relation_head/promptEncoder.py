import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.nn.init import xavier_normal_


class PromptEncoder(torch.nn.Module):
    def __init__(self, config):
        super(PromptEncoder, self).__init__()
        self.cfg = config
        self.n_layer = self.cfg.MODEL.PROMPTENCODER.N_RELATION_ENCODER_LAYER  #params.n_relation_encoder_layer #3  
        self.hidden_dim = self.cfg.MODEL.PROMPTENCODER.HIDDEN_DIM #params.hidden_dim #32
        self.attn_dim = self.cfg.MODEL.PROMPTENCODER.ATT_DIM #params.attn_dim #5

        # initialize embeddings
        self.start_relation_embeddings = nn.Embedding(1, self.hidden_dim)
        self.position_embedding = nn.Embedding((self.cfg.MODEL.PROMPTENCODER.PATH_HOP+1)*(self.cfg.MODEL.PROMPTENCODER.PATH_HOP+1), self.hidden_dim) #path_hop=3
        self.self_loop_embedding = nn.Embedding(1, self.hidden_dim)
        xavier_normal_(self.start_relation_embeddings.weight.data)
        xavier_normal_(self.position_embedding.weight.data)
        xavier_normal_(self.self_loop_embedding.weight.data)

        self.act = nn.RReLU()

        self.W_ht2r = nn.ModuleList([nn.Linear(self.hidden_dim * 3, self.hidden_dim, bias=True) for _ in range(self.n_layer)])
        self.W_message = nn.ModuleList([nn.Linear(self.hidden_dim * 3 if self.cfg.MODEL.PROMPTENCODER.MSG == 'concat' else self.hidden_dim * 5, self.hidden_dim, bias=True) for _ in range(self.n_layer)])
        self.alpha = nn.ModuleList([nn.Linear(self.hidden_dim * 2, 1, bias=True) for _ in range(self.n_layer)])
        self.beta = nn.ModuleList([nn.Linear(self.hidden_dim * 2, 1, bias=True) for _ in range(self.n_layer)])
        self.loop_transfer = nn.ModuleList([nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=True) for _ in range(self.n_layer)])
        self.ent_transfer = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim, bias=True) for _ in range(self.n_layer)])
        self.rel_transfer = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim, bias=True) for _ in range(self.n_layer)])

        # readout
        self.final_to_rel_embeddings = nn.Linear(self.hidden_dim * self.n_layer, self.hidden_dim)

        # dropout and layer normalization
        self.dropout = nn.Dropout(0.3)
        self.layer_norm_rels = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layer+1)])
        self.layer_norm_ents = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layer+1)])
        self.layer_norm_loop = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layer+1)])
        # self.gate_rel = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, edge_index, edge_type, h_positions, t_positions, query_relations, edge_query_relations, labels, num_ent, loader, shot=5):
        relation_num = loader.kg.relation_num #当前kg中关系的数量
        final_rel_embeddings = []

        if self.cfg.MODEL.PROMPTENCODER.USE_TOKEN_SET: #True
            # initialize entity embeddings
            position = labels[:, 0] * (self.cfg.MODEL.PROMPTENCODER.PATH_HOP+1) + labels[:, 1]  #等于label数 [num_labels]  后面会作为node_emb的索引
            node_embeddings = torch.index_select(self.position_embedding.weight, 0, position) # [num_ent, hidden_dim]
            node_embeddings[h_positions] = self.position_embedding.weight[0]  #因为query的head 自己是头节点，所以到自己的距离是0
            node_embeddings[t_positions] = self.position_embedding.weight[1]  #同理，tail到头节点的距离是1

            # initialize relation embeddings
            rel_embeddings = torch.zeros(relation_num * h_positions.size(-1), self.hidden_dim).cuda() # [num_rel x num_fact, 32]  表示每个关系在每个事实中的嵌入。
            rel_embeddings[query_relations+torch.arange(query_relations.size(-1)).cuda()*relation_num] = self.start_relation_embeddings.weight[0] 
        else:
            # initialize entity and relation embeddings (w/o unified token set)
            node_embeddings = torch.zeros(num_ent, self.hidden_dim).cuda()
            rel_embeddings = torch.zeros(relation_num * h_positions.size(-1), self.hidden_dim).cuda()
            nn.init.xavier_uniform_(node_embeddings)
            nn.init.xavier_uniform_(rel_embeddings)

        self_loop_embeddings = self.self_loop_embedding.weight.unsqueeze(0).expand(  #[num_fact,32]
            rel_embeddings.size(0) // relation_num, -1, -1).reshape(-1, self.hidden_dim)  #为所有边设置一个相同的自环嵌入，重塑成与rel_emb相同的形状

        # prompt encoder
        for i in range(self.n_layer):
            # 1. update node embeddings
            r_embeddings = torch.index_select(rel_embeddings, 0, edge_type)  #[num_edge,32]
            h_embeddings = torch.index_select(node_embeddings, 0, edge_index[0])
            q_embeddings = torch.index_select(rel_embeddings, 0, edge_query_relations) #edge_query_relations：整个图的边（加上偏移量后）

            # 1.1 message with attention
            feature = self.entity_message(h_embeddings, r_embeddings, q_embeddings, self.cfg.MODEL.PROMPTENCODER.MSG) # 生成尾实体特征
            message = self.act(self.W_message[i](feature))
            # alpha = self.entity_attention(torch.cat([r_embeddings, q_embeddings], dim=-1), edge_index,
            #                               node_embeddings.size(0), i)
            alpha = self.entity_attention(torch.cat([r_embeddings, q_embeddings], dim=-1), edge_index, node_embeddings.size(0), i) #1维度cat
            message = message * alpha
            ent_norm = self.compute_norm(edge_index, num_ent)  #[num_rel] 计算图中每条边的归一化权重。它通过节点的度的逆平方根对边权重进行归一化
            message = message * ent_norm.view(-1, 1)

            # 1.2 aggregation
            node_embeddings = self.ent_aggregation(message, edge_index, num_ent, self.cfg.MODEL.PROMPTENCODER.AGG, i)

            # 1.3 layer normalization
            node_embeddings = self.layer_norm_ents[i](node_embeddings) #[num_node,32]

            # 2. update relation embeddings
            h_embeddings = torch.index_select(node_embeddings, 0, edge_index[0])
            t_embeddings = torch.index_select(node_embeddings, 0, edge_index[1])
            r_embeddings = torch.index_select(rel_embeddings, 0, edge_type)

            # 2.1 message with attention
            feature = torch.cat([h_embeddings, t_embeddings, q_embeddings], dim=1)
            message = self.act(self.W_ht2r[i](feature))
            beta = self.relation_attention(torch.cat([r_embeddings, q_embeddings], dim=-1), edge_type,
                                           rel_embeddings.size(0), i)
            message = message * beta
            # 2.2 aggregation
            rel_embeddings = self.rel_aggregation(rel_embeddings, message, edge_type, rel_embeddings.size(0), i)
            # 2.3 layer normalization
            rel_embeddings = self.layer_norm_rels[i](rel_embeddings)

            # 3. store the relation embeddings of this layer
            final_rel_embeddings.append(rel_embeddings)

            # 4. update self-loop embeddings
            qr_embeddings = torch.index_select(rel_embeddings, 0, query_relations + torch.arange( #从关系嵌入 rel_embeddings 中选择与查询关系 query_relations 对应的嵌入
                query_relations.size(-1)).cuda() * relation_num) #[num_fact,32]
            self_loop_embeddings = self_loop_embeddings + self.act(  #[num_fact,32]
                self.loop_transfer[i](torch.cat([self_loop_embeddings, qr_embeddings], dim=-1))) #将自环边嵌入与查询关系嵌入拼接，并通过线性变换和激活函数进行增强
            self_loop_embeddings = self.layer_norm_loop[i](self_loop_embeddings)  

        # readout
        final_rel_embeddings = torch.cat(final_rel_embeddings, dim=-1)  #三层rel_emb 横着cat
        final_rel_embeddings = self.act(self.final_to_rel_embeddings(final_rel_embeddings))
        final_rel_embeddings = self.layer_norm_rels[-1](final_rel_embeddings)
        final_rel_embeddings = final_rel_embeddings.view(-1, shot, relation_num,  self.hidden_dim)  #[num_unique_rel,5,num_rel,32]

        # if multi-shot, then calculate the average of the final relation embeddings
        final_rel_embeddings_full = final_rel_embeddings
        final_rel_embeddings = torch.mean(final_rel_embeddings, dim=1).view(-1, relation_num, self.hidden_dim)  #[num_unique_rel,num_rel,32]  把每个关系对应的五个case的rel_emb均值化
        self_loop_embeddings = torch.mean(self_loop_embeddings.view(-1, shot, 1, self.hidden_dim), dim=1).view(-1, 1, self.hidden_dim)  #[num_unique_rel,1,32]  每个唯一关系的自环表示
        final_rel_embeddings = self.dropout(final_rel_embeddings)

        final_rel_embeddings = final_rel_embeddings.view(-1, relation_num, self.hidden_dim)
        final_rel_embeddings = torch.cat([final_rel_embeddings, self_loop_embeddings], dim=1)
        return final_rel_embeddings, None, final_rel_embeddings_full

    def entity_message(self, h, r, q, MSG):
        if MSG == 'add':
            feature = h + r
        elif MSG == 'mul':
            feature = h * r
        elif MSG == 'concat':
            feature = torch.cat([h, r, q], dim=-1)
        elif MSG == 'mix':
            feature = torch.cat([h * r, h + r, h, r, q], dim=-1)
        else:
            raise NotImplementedError
        return feature

    def entity_attention(self, feature, edge_index, num_nodes, i):
        if self.cfg.MODEL.PROMPTENCODER.USE_ATT:  # for ablation study True
            alpha = self.alpha[i](self.act(feature))
            if self.cfg.MODEL.PROMPTENCODER.ATT_TYPE == 'GAT':
                alpha = torch.exp(alpha)
                alpha = alpha / (torch.index_select(
                    scatter_add(alpha, edge_index[1], dim=0, dim_size=num_nodes)[edge_index[1]] + 1e-10,
                    0, edge_index[1]))
            elif self.cfg.MODEL.PROMPTENCODER.ATT_TYPE == 'Sigmoid':
                alpha = torch.sigmoid(alpha)
        else:
            alpha = 1.0
        return alpha

    def ent_aggregation(self, message, edge_index, num_ent, AGG, i):
        ent_norm = self.compute_norm(edge_index, num_ent)
        message = message * ent_norm.view(-1, 1)
        if AGG == 'sum':
            node_embeddings_ = scatter_add(message, index=edge_index[1], dim=0, dim_size=num_ent)
        elif AGG == 'max':
            node_embeddings_, _ = scatter_max(message, index=edge_index[1], dim=0, dim_size=num_ent)
        elif AGG == 'mean':
            node_embeddings_ = scatter_mean(message, index=edge_index[1], dim=0, dim_size=num_ent)
        else:
            raise NotImplementedError
        node_embeddings_ = self.act(self.ent_transfer[i](node_embeddings_))
        return node_embeddings_

    def relation_attention(self, feature, edge_type, num_rels, i):
        if self.cfg.MODEL.PROMPTENCODER.USE_ATT: #True
            beta = self.beta[i](feature)
            if self.cfg.MODEL.PROMPTENCODER.ATT_TYPE == 'GAT':
                beta = torch.exp(beta)
                beta = beta / (torch.index_select(
                    scatter_add(beta, index=edge_type, dim=0, dim_size=num_rels)[edge_type] + 1e-10, 0,edge_type))
            elif self.cfg.MODEL.PROMPTENCODER.ATT_TYPE == 'Sigmoid':
                beta = torch.sigmoid(beta)
            else:
                raise NotImplementedError
        else:
            beta = 1.0
        return beta

    def rel_aggregation(self, rel_embeddings, message, edge_type, num_rels, i):
        if self.cfg.MODEL.PROMPTENCODER.AGG_REL == 'max':
            rel_embeddings_, _ = scatter_max(message, index=edge_type, dim=0,
                                             dim_size=num_rels)
        elif self.cfg.MODEL.PROMPTENCODER.AGG_REL == 'sum':
            rel_embeddings_ = scatter_add(message, index=edge_type, dim=0,
                                          dim_size=num_rels)
        elif self.cfg.MODEL.PROMPTENCODER.AGG_REL == 'mean':
            rel_embeddings_ = scatter_mean(message, index=edge_type, dim=0,
                                           dim_size=num_rels)
        else:
            raise NotImplementedError
        rel_embeddings_ = self.act(self.rel_transfer[i](rel_embeddings_))
        rel_embeddings = rel_embeddings_ + rel_embeddings
        rel_embeddings = rel_embeddings.squeeze(0)
        return rel_embeddings

    def compute_norm(self, edge_index, num_ent):
        col, row = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}
        return norm