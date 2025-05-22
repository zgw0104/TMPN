import torch
from math import sqrt

from torch import nn
from torch.nn import functional as F
import numpy as np

from hetsgg.config import cfg
from hetsgg.data import get_dataset_statistics
from hetsgg.modeling import registry
from hetsgg.modeling.roi_heads.relation_head.classifier import build_classifier
from hetsgg.modeling.roi_heads.relation_head.model_kern import (
    to_onehot,
)

from hetsgg.modeling.utils import cat
from hetsgg.structures.boxlist_ops import squeeze_tensor
from .model_motifs import FrequencyBias

from hetsgg.modeling.roi_heads.relation_head.model_HetSGG import HetSGG
from hetsgg.modeling.roi_heads.relation_head.model_HetSGGplus import HetSGGplus_Context

from .rel_proposal_network.loss import (
    RelAwareLoss,
)
from .utils_relation import obj_prediction_nms
from .utils_motifs import obj_edge_vectors, rel_vectors, encode_box_info
from .utils_relation import layer_init, get_box_info, get_box_pair_info, obj_prediction_nms
from hetsgg.modeling.make_layers import make_fc
import h5py
from .utils_relation import nms_overlaps

from .data_loader import *

from .promptEncoder import PromptEncoder


@registry.ROI_RELATION_PREDICTOR.register("PrototypeEmbeddingNetwork")
class PrototypeEmbeddingNetwork(nn.Module):
    def __init__(self, config, in_channels):
        super(PrototypeEmbeddingNetwork, self).__init__()

        self.num_obj_cls = 151#config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = 51#config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels
        

        #self.use_vision = config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        
        self.hidden_dim = 512#config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 
        self.pooling_dim = 4096#config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048 # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)  

        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2 # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT
        
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
       
        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim*2, self.mlp_dim)  
        self.gate_obj = nn.Linear(self.mlp_dim*2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim*2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim*2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
       
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2) 

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes) 
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        
        self.nms_thresh = 0.5#self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        self.post_cat = MLP(self.mlp_dim, self.mlp_dim * 2, self.mlp_dim, 2)
        # layer_init(self.post_cat, xavier=True)

        proto = '/root/het/proto/muil_sgl_p.h5'

        f = h5py.File(proto, 'r')

        # for p_0
        self.vis_proto_0 = torch.from_numpy(f['proto_0'][:])
        self.vis_W_0 = MLP(self.mlp_dim, self.mlp_dim * 2, self.mlp_dim, 2)
        self.norm_vis_rep_0 = nn.LayerNorm(self.mlp_dim)
        self.dropout_vis_rep_0 = nn.Dropout(dropout_p)
        self.linear_vis_rep_0 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.dropout_vis_0 = nn.Dropout(dropout_p)
        self.project_head_vis_0 = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)
        self.dropout_pred_vis_0 = nn.Dropout(dropout_p)
        self.logit_scale_vis_0 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.vis_embed_0 = nn.Embedding(51,2048)
        with torch.no_grad():
            self.vis_embed_0.weight.copy_(self.vis_proto_0, non_blocking=True)

        # end

        # for p_1
        self.vis_proto_1 = torch.from_numpy(f['proto_1'][:])
        self.vis_W_1 = MLP(self.mlp_dim, self.mlp_dim * 2, self.mlp_dim, 2)
        self.norm_vis_rep_1 = nn.LayerNorm(self.mlp_dim)
        self.dropout_vis_rep_1 = nn.Dropout(dropout_p)
        self.linear_vis_rep_1 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.dropout_vis_1 = nn.Dropout(dropout_p)
        self.project_head_vis_1 = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)
        self.dropout_pred_vis_1 = nn.Dropout(dropout_p)
        self.logit_scale_vis_1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.vis_embed_1 = nn.Embedding(51,2048)
        with torch.no_grad():
            self.vis_embed_1.weight.copy_(self.vis_proto_1, non_blocking=True)

        # end

        # for p_2
        self.vis_proto_2 = torch.from_numpy(f['proto_2'][:])
        self.vis_W_2 = MLP(self.mlp_dim, self.mlp_dim * 2, self.mlp_dim, 2)
        self.norm_vis_rep_2 = nn.LayerNorm(self.mlp_dim)
        self.dropout_vis_rep_2 = nn.Dropout(dropout_p)
        self.linear_vis_rep_2 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.dropout_vis_2 = nn.Dropout(dropout_p)
        self.project_head_vis_2 = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)
        self.dropout_pred_vis_2 = nn.Dropout(dropout_p)
        self.logit_scale_vis_2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.vis_embed_2 = nn.Embedding(51,2048)
        with torch.no_grad():
            self.vis_embed_2.weight.copy_(self.vis_proto_2, non_blocking=True)

        layer_init(self.linear_vis_rep_0,xavier=True)
        layer_init(self.linear_vis_rep_1,xavier=True)
        layer_init(self.linear_vis_rep_2,xavier=True)

        # layer_init(self.vis_W_0,xavier=True)
        # layer_init(self.vis_W_1,xavier=True)
        # layer_init(self.vis_W_2,xavier=True)

        # layer_init(self.project_head_vis_0,xavier=True)
        # layer_init(self.project_head_vis_1,xavier=True)
        # layer_init(self.project_head_vis_2,xavier=True)

        self.context_layer = HetSGG(config, in_channels)
        self.ent_post = MLP(128,1024,4096,2)
        self.rel_post = MLP(128,1024,2048,2)

        ### het
        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS
        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS
        self.n_reltypes = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.NUM_RELATION
        self.n_dim = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.H_DIM
        self.n_ntypes = int(sqrt(self.n_reltypes))
        self.obj2rtype = {(i, j): self.n_ntypes*j+i for j in range(self.n_ntypes) for i in range(self.n_ntypes)}
        self.rel_aware_model_on = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS
        self.rel_classifier = build_classifier(self.num_rel_cls, self.num_rel_cls) # Linear Layer
        self.obj_classifier = build_classifier(self.num_obj_cls, self.num_obj_cls)

        assert in_channels is not None
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)
            self.freq_lambda = nn.Parameter(
                torch.Tensor([1.0]), requires_grad=False
            )  # hurt performance when set learnable

        ### for prompt
        pt_path = '/root/p_data/vg/train'
        self.data_loader = [DataLoader(data_dir= pt_path, dataset = pt_path.split('/')[-2] )]
        self.relation_encoder = PromptEncoder(self.cfg)
        self.relation_num = 102 # 包括自环边
        self.prompt_hidden_dim = 32
        self.for_zero_rel_forward = nn.Embedding(51,32)
        self.prompt_linear = nn.Linear(self.mlp_dim + config.MODEL.PROMPTENCODER.HIDDEN_DIM * 2, self.mlp_dim)

        self.vis_proto_p = torch.from_numpy(f['proto_0'][:])
        self.vis_W_p = MLP(self.mlp_dim, self.mlp_dim * 2, self.mlp_dim, 2)
        self.norm_vis_rep_p = nn.LayerNorm(self.mlp_dim)
        self.dropout_vis_rep_p = nn.Dropout(dropout_p)
        self.linear_vis_rep_p = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.dropout_vis_p = nn.Dropout(dropout_p)
        self.project_head_vis_p = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)
        self.dropout_pred_vis_p = nn.Dropout(dropout_p)
        self.logit_scale_vis_p = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.vis_embed_p = nn.Embedding(51,2048)
        with torch.no_grad():
            self.vis_embed_p.weight.copy_(self.vis_proto_0, non_blocking=True)

        layer_init(self.linear_vis_rep_p,xavier=True)
        layer_init(self.prompt_linear,xavier=True)

        

    def init_classifier_weight(self):
        self.obj_classifier.reset_parameters()
        for i in self.n_reltypes:
            self.rel_classifier[i].reset_parameters()

        

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, is_training=True):

        add_losses = {}
        add_data = {}


        

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        ##### 

        ent_feat, rel_feat = self.context_layer(roi_features, union_features, proposals, rel_pair_idxs, rel_binarys, logger, is_training)

        rel_feat = self.rel_post(rel_feat)
        # entity_rep = self.ent_post(ent_feat)
        entity_rep = self.post_emb(roi_features)   # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)    # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)    # xo

        entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe 

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps, entity_preds, entity_embeds, proposals):

            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  #  Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  #  Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)
            
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj)) # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)  
        fusion_so = self.post_cat(fusion_so)
        # fusion_so = self.rel_post(rel_feat)
        # pair_pred = cat(pair_preds, dim=0) 

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        #vis_prototpye_0  use het rel_rep
        vis_rep_x = rel_feat
        
        vis_proto_0 = self.vis_W_0(self.vis_embed_0.weight)

        #vis model convergence
        vis_rep_0 = self.norm_vis_rep_0(self.dropout_vis_rep_0(torch.relu(self.linear_vis_rep_0(vis_rep_x))) + vis_rep_x)
        vis_rep_0 = self.project_head_vis_0(self.dropout_vis_0(torch.relu(vis_rep_0)))
        predicate_proto_vis_0 = self.project_head_vis_0(self.dropout_pred_vis_0(torch.relu(vis_proto_0)))

        vis_rep_norm_0 = vis_rep_0 / vis_rep_0.norm(dim=1, keepdim=True)
        predicate_proto_vis_norm_0 = predicate_proto_vis_0 / predicate_proto_vis_0.norm(dim=1, keepdim=True)

        vis_dist_0 = vis_rep_norm_0 @ predicate_proto_vis_norm_0.t() * self.logit_scale_vis_0.exp()
        

        # end

        #vis_prototpye_1
        vis_rep = fusion_so
        vis_proto_1 = self.vis_W_1(self.vis_embed_1.weight)

        #vis model convergence
        vis_rep_1 = self.norm_vis_rep_1(self.dropout_vis_rep_1(torch.relu(self.linear_vis_rep_1(vis_rep))) + vis_rep)
        vis_rep_1 = self.project_head_vis_1(self.dropout_vis_1(torch.relu(vis_rep_1)))
        predicate_proto_vis_1 = self.project_head_vis_1(self.dropout_pred_vis_1(torch.relu(vis_proto_1)))

        vis_rep_norm_1 = vis_rep_1 / vis_rep_1.norm(dim=1, keepdim=True)
        predicate_proto_vis_norm_1 = predicate_proto_vis_1 / predicate_proto_vis_1.norm(dim=1, keepdim=True)

        vis_dist_1 = vis_rep_norm_1 @ predicate_proto_vis_norm_1.t() * self.logit_scale_vis_1.exp()
        

        # end

        #vis_prototpye_2
        vis_proto_2 = self.vis_W_2(self.vis_embed_2.weight)

        #vis model convergence
        vis_rep_2 = self.norm_vis_rep_2(self.dropout_vis_rep_2(torch.relu(self.linear_vis_rep_2(vis_rep))) + vis_rep)
        vis_rep_2 = self.project_head_vis_2(self.dropout_vis_2(torch.relu(vis_rep_2)))
        predicate_proto_vis_2 = self.project_head_vis_2(self.dropout_pred_vis_2(torch.relu(vis_proto_2)))

        vis_rep_norm_2 = vis_rep_2 / vis_rep_2.norm(dim=1, keepdim=True)
        predicate_proto_vis_norm_2 = predicate_proto_vis_2 / predicate_proto_vis_2.norm(dim=1, keepdim=True)

        vis_dist_2 = vis_rep_norm_2 @ predicate_proto_vis_norm_2.t() * self.logit_scale_vis_2.exp()
        

        # end

        avg_vis_proto_dist = (vis_dist_0 + vis_dist_1 + vis_dist_2) / 3









        rel_rep = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        
        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        ######

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        rel_dists = rel_dists + avg_vis_proto_dist


        _, rel_list = torch.max(rel_dists,1)

        if self.training:
            rel = rel_labels
            unique_triplet = self.get_triplet(proposals, rel_pair_idxs, rel)

        else:
            rel = rel_list
            unique_triplet = self.get_triplet2(proposals, rel_pair_idxs, rel)
        
        
        rel_emb_forward, rel_emb_inverse, unique_rels = self.prompt_vec(unique_triplet)

        #vis_prototpye_0  use het rel_rep
        # vis_rep_p = rel_feat

        #for prompt
        fusion_prompt = rel_feat
        if self.training:
            r_labels = cat(rel,dim=0)
        else:
            r_labels = rel
        u_rel = unique_rels.tolist()
        u_rel.insert(0,0)
        #construct prompt vec
        u_rel_idx = []
        for i in range(r_labels.shape[0]):
            u_rel_idx.append(u_rel.index(r_labels[i]))
        prompt_for, prompt_inv = rel_emb_forward[u_rel_idx,r_labels,:], rel_emb_inverse[u_rel_idx,r_labels,:]
        fusion_prompt = cat((prompt_for,fusion_prompt), dim=1)
        fusion_prompt = cat((fusion_prompt,prompt_inv), dim=1)

        vis_rep_p = fusion_prompt
        vis_rep_x = self.prompt_linear(vis_rep_x)
        
        vis_proto_p = self.vis_W_0(self.vis_embed_p.weight)

        #vis model convergence
        vis_rep_p = self.norm_vis_rep_p(self.dropout_vis_rep_p(torch.relu(self.linear_vis_rep_p(vis_rep_p))) + vis_rep_p)
        vis_rep_p = self.project_head_vis_p(self.dropout_vis_p(torch.relu(vis_rep_p)))
        predicate_proto_vis_p = self.project_head_vis_p(self.dropout_pred_vis_p(torch.relu(vis_proto_p)))

        vis_rep_norm_p = vis_rep_p / vis_rep_p.norm(dim=1, keepdim=True)
        predicate_proto_vis_norm_p = predicate_proto_vis_p / predicate_proto_vis_p.norm(dim=1, keepdim=True)

        vis_dist_p = vis_rep_norm_p @ predicate_proto_vis_norm_p.t() * self.logit_scale_vis_p.exp()

        rel_dists = rel_dists + vis_dist_p * 0.1

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.training:

            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)    #p=2 2范数，即平方和开根， p=1 1范数，即绝对值求和
            add_losses.update({"l21_loss": l21 * 0.8})  # Le_sim = ||S||_{2,1}
            ### end
            
            ### Vis Prototype Regularization (0)----cosine similarity 
            target_vis_predicate_proto_norm_0 = predicate_proto_vis_norm_0.clone().detach()
            simil_mat_vis_0 = predicate_proto_vis_norm_0 @ target_vis_predicate_proto_norm_0.t()
            l21_vis_0 = torch.norm(torch.norm(simil_mat_vis_0, p=2, dim=1), p=1) / (51*51)
            add_losses.update({"l21_vis_loss_0":l21_vis_0 * 0.2})

            ### end

            ### Vis Prototype Regularization (1)----cosine similarity 
            target_vis_predicate_proto_norm_1 = predicate_proto_vis_norm_1.clone().detach()
            simil_mat_vis_1 = predicate_proto_vis_norm_1 @ target_vis_predicate_proto_norm_1.t()
            l21_vis_1 = torch.norm(torch.norm(simil_mat_vis_1, p=2, dim=1), p=1) / (51*51)
            add_losses.update({"l21_vis_loss_1":l21_vis_1 * 0.2})

            ### end

            ### Vis Prototype Regularization (2)----cosine similarity 
            target_vis_predicate_proto_norm_2 = predicate_proto_vis_norm_2.clone().detach()
            simil_mat_vis_2 = predicate_proto_vis_norm_2 @ target_vis_predicate_proto_norm_2.t()
            l21_vis_2 = torch.norm(torch.norm(simil_mat_vis_2, p=2, dim=1), p=1) / (51*51)
            add_losses.update({"l21_vis_loss_2":l21_vis_2 * 0.2})

            ### end







            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, 51, -1) # pre_proto=[51,4096].unsqueeze(dim=1)->[51,1,4096].expand(-1,51,-1)->[51,51,4096] 每行复制成[51,4096]然后复制51张
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(51, -1, -1) # pre_proto=[51,4096].unsqueeze(dim=0)->[1,51,4096].expand(51,-1,-1) ->[51,51,4096] 直接复制51张
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2  对角线=0 其余位置为 本行元素(原型本身)与本列元素(其它原型)的距离  [51,51]
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)   # dim=1  按行来排序， sort返回两个值，一个是排序后的矩阵一个是元素索引矩阵
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1   只取最近的一个原型   sum(dim=1) 按行求和
            dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss * 0.8})
            ### end 

            ### Vis Prototype Regularization (0) ---- Euclidean distance
            gamma3_0 = 7.0
            vis_predicate_proto_a_0 = vis_proto_0.unsqueeze(dim=1).expand(-1,51,-1)
            vis_predicate_proto_b_0 = vis_proto_0.detach().unsqueeze(dim=0).expand(51,-1,-1)
            vis_proto_dis_mat_0 = (vis_predicate_proto_a_0 - vis_predicate_proto_b_0).norm(dim=2) ** 2
            sorted_vis_proto_dis_mat_0, _ = torch.sort(vis_proto_dis_mat_0, dim=1)
            topK_proto_dis_vis_0 = sorted_vis_proto_dis_mat_0[:, :2].sum(dim=1) / 1
            dist_loss_vis_0 = torch.max(torch.zeros(51).cuda(), -topK_proto_dis_vis_0 + gamma3_0).mean()
            add_losses.update({"dist_loss_vis_0": dist_loss_vis_0 * 0.2})
            ### end

            ### Vis Prototype Regularization (1) ---- Euclidean distance
            gamma3_1 = 7.0
            vis_predicate_proto_a_1 = vis_proto_1.unsqueeze(dim=1).expand(-1,51,-1)
            vis_predicate_proto_b_1 = vis_proto_1.detach().unsqueeze(dim=0).expand(51,-1,-1)
            vis_proto_dis_mat_1 = (vis_predicate_proto_a_1 - vis_predicate_proto_b_1).norm(dim=2) ** 2
            sorted_vis_proto_dis_mat_1, _ = torch.sort(vis_proto_dis_mat_1, dim=1)
            topK_proto_dis_vis_1 = sorted_vis_proto_dis_mat_1[:, :2].sum(dim=1) / 1
            dist_loss_vis_1 = torch.max(torch.zeros(51).cuda(), -topK_proto_dis_vis_1 + gamma3_1).mean()
            add_losses.update({"dist_loss_vis_1": dist_loss_vis_1 * 0.2})
            ### end

            ### Vis Prototype Regularization (2) ---- Euclidean distance
            gamma3_2 = 7.0
            vis_predicate_proto_a_2 = vis_proto_2.unsqueeze(dim=1).expand(-1,51,-1)
            vis_predicate_proto_b_2 = vis_proto_2.detach().unsqueeze(dim=0).expand(51,-1,-1)
            vis_proto_dis_mat_2 = (vis_predicate_proto_a_2 - vis_predicate_proto_b_2).norm(dim=2) ** 2
            sorted_vis_proto_dis_mat_2, _ = torch.sort(vis_proto_dis_mat_2, dim=1)
            topK_proto_dis_vis_2 = sorted_vis_proto_dis_mat_2[:, :2].sum(dim=1) / 1
            dist_loss_vis_2 = torch.max(torch.zeros(51).cuda(), -topK_proto_dis_vis_2 + gamma3_2).mean()
            add_losses.update({"dist_loss_vis_2": dist_loss_vis_2 * 0.2})
            ### end
            

            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, 51, -1)  # r  r=[842,4096].unsqueeze(dim=1) -> [842,1,4096].expand(-1,51,-1) -> [842,51,4096] 先将每行表征提出然后复制成51份 然后和原型比较
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2    [y,51] y=sum(num_rel)

            mask_neg = torch.ones(rel_labels.size(0), 51).cuda()  
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+

            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum * 0.8 })     # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end 

            ### Prototype-based Learning for Vis (0)---- Euclidean distance
            gamma1_0 = 1.0
            vis_rep_expand_0 = vis_rep_0.unsqueeze(dim=1).expand(-1,51,-1)
            predicate_proto_expand_0 = predicate_proto_vis_0.unsqueeze(dim=0).expand(rel_labels.size(0),-1,-1)
            distance_set_0 = (vis_rep_expand_0 - predicate_proto_expand_0).norm(dim=2) ** 2
            mask_neg_0 = torch.ones(rel_labels.size(0), 51).cuda()
            mask_neg_0[torch.arange(rel_labels.size(0)),rel_labels] = 0
            distance_set_neg_0 = distance_set_0 * mask_neg_0
            distance_set_pos_0 = distance_set_0[torch.arange(rel_labels.size(0)),rel_labels]

            sorted_distance_set_neg_0, _ = torch.sort(distance_set_neg_0,dim=1)
            topK_sorted_distance_set_neg_0 = sorted_distance_set_neg_0[:, :11].sum(dim=1) / 10
            loss_sum_0 = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos_0 - topK_sorted_distance_set_neg_0 + gamma1_0).mean()
            add_losses.update({"loss_dis_vis_0": loss_sum_0 * 0.2})
            ### end

            ### Prototype-based Learning for Vis (1)---- Euclidean distance
            gamma1_1 = 1.0
            vis_rep_expand_1 = vis_rep_1.unsqueeze(dim=1).expand(-1,51,-1)
            predicate_proto_expand_1 = predicate_proto_vis_1.unsqueeze(dim=0).expand(rel_labels.size(0),-1,-1)
            distance_set_1 = (vis_rep_expand_1 - predicate_proto_expand_1).norm(dim=2) ** 2
            mask_neg_1 = torch.ones(rel_labels.size(0), 51).cuda()
            mask_neg_1[torch.arange(rel_labels.size(0)),rel_labels] = 0
            distance_set_neg_1 = distance_set_1 * mask_neg_1
            distance_set_pos_1 = distance_set_1[torch.arange(rel_labels.size(0)),rel_labels]

            sorted_distance_set_neg_1, _ = torch.sort(distance_set_neg_1,dim=1)
            topK_sorted_distance_set_neg_1 = sorted_distance_set_neg_1[:, :11].sum(dim=1) / 10
            loss_sum_1 = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos_1 - topK_sorted_distance_set_neg_1 + gamma1_1).mean()
            add_losses.update({"loss_dis_vis_1": loss_sum_1 * 0.2})
            ### end

            ### Prototype-based Learning for Vis (2)---- Euclidean distance
            gamma1_2 = 1.0
            vis_rep_expand_2 = vis_rep_2.unsqueeze(dim=1).expand(-1,51,-1)
            predicate_proto_expand_2 = predicate_proto_vis_2.unsqueeze(dim=0).expand(rel_labels.size(0),-1,-1)
            distance_set_2 = (vis_rep_expand_2 - predicate_proto_expand_2).norm(dim=2) ** 2
            mask_neg_2 = torch.ones(rel_labels.size(0), 51).cuda()
            mask_neg_2[torch.arange(rel_labels.size(0)),rel_labels] = 0
            distance_set_neg_2 = distance_set_2 * mask_neg_2
            distance_set_pos_2 = distance_set_2[torch.arange(rel_labels.size(0)),rel_labels]

            sorted_distance_set_neg_2, _ = torch.sort(distance_set_neg_2,dim=1)
            topK_sorted_distance_set_neg_2 = sorted_distance_set_neg_2[:, :11].sum(dim=1) / 10
            loss_sum_2 = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos_2 - topK_sorted_distance_set_neg_2 + gamma1_2).mean()
            add_losses.update({"loss_dis_vis_2": loss_sum_2 * 0.2})
            ### end
 
        return entity_dists, rel_dists, add_losses #, add_data



    def prompt_vec(self, unique_triplet):
        unique_rels = unique_triplet[:,1].cpu().numpy()
        unique_rels, unique_indices = np.unique(unique_rels, return_inverse=True)
        
        repeat_rels = np.repeat(np.expand_dims(unique_rels, 1), self.cfg.MODEL.PROMPTENCODER.SHOT, axis=1).reshape(-1) #把唯一关系重复5次  5x

        edge_index, edge_type, h_positions, t_positions, query_relations, edge_query_relations, labels, num_ent = self.data_loader[0].get_case_graph(repeat_rels #导入整个数据集的图，x_position是query头尾节点的在提取出的子图中的索引
                                                                                                                     )            
        rel_embeddings, query_embeddings, rel_embedding_full = self.relation_encoder(edge_index, edge_type, h_positions, t_positions, query_relations, edge_query_relations, labels,
                                                   num_ent, self.data_loader[0], shot=self.cfg.MODEL.PROMPTENCODER.SHOT)
        final_rel_embeddings = torch.mean(rel_embedding_full, dim=1).view(-1, self.relation_num, self.prompt_hidden_dim) #[num_unique_rel,102,32]
        #1、把unique_rel 对应的整组emb求均值，再把每个case的求均值

        #2、把反转关系和正关系做融合
        rel_emb_forward, rel_emb_inverse = torch.split(final_rel_embeddings, 51, dim=1)
        m = self.for_zero_rel_forward.weight
        m = m.unsqueeze(0)
        rel_emb_forward = cat((m,rel_emb_forward), dim=0)
        rel_emb_inverse = cat((m,rel_emb_inverse), dim=0)

        return rel_emb_forward, rel_emb_inverse, unique_rels

    def find_label(self, label, rel_idx):
        rel = rel_idx.clone().detach()
        for i in range(rel_idx.shape[0]):
            rel[i,0] = label[rel_idx[i,0].item()].item()
            rel[i,1] = label[rel_idx[i,1].item()].item()
        return rel
    
    def get_triplet(self, proposals, rel_pair_idx, rel_labels):
        rel = self.find_label(proposals[0].get_field("labels"),rel_pair_idx[0])
        edge_idx = rel_labels[0]
        no_zero_indice = torch.nonzero(edge_idx).squeeze(dim=1)
        edge_idx = edge_idx.unsqueeze(dim=0).t()
        triplet_head = rel[:,:1]
        triplet_tail = rel[:,1:]
        triplet = cat((triplet_head,edge_idx),dim=1)
        triplet = cat((triplet,triplet_tail),dim=1)
        triplet = triplet[no_zero_indice,:]

        for i in range(1,len(proposals)):
            rel_p = self.find_label(proposals[i].get_field("labels"), rel_pair_idx[i])
            edge_idx_ = rel_labels[i]
            no_zero_indice_ = torch.nonzero(edge_idx_).squeeze(dim=1)
            edge_idx_ = edge_idx_.unsqueeze(dim=0).t()
            triplet_head_ = rel_p[:,:1]
            triplet_tail_ = rel_p[:,1:]
            triplet_ = cat((triplet_head_,edge_idx_),dim=1)
            triplet_ = cat((triplet_,triplet_tail_),dim=1)
            triplet_ = triplet_[no_zero_indice_,:]

            triplet = cat((triplet,triplet_),dim=0)

        unique_triplet = torch.unique(triplet,dim=0)
        return unique_triplet

    def get_triplet2(self, proposals, rel_pair_idx, rel_labels):
        rel = self.find_label(proposals[0].get_field("labels"),rel_pair_idx[0])
        edge_idx = rel_labels
        no_zero_indice = torch.nonzero(edge_idx).squeeze(dim=1)
        edge_idx = edge_idx.unsqueeze(dim=0).t()
        triplet_head = rel[:,:1]
        triplet_tail = rel[:,1:]
        triplet = cat((triplet_head,edge_idx),dim=1)
        triplet = cat((triplet,triplet_tail),dim=1)
        triplet = triplet[no_zero_indice,:]
        # first = rel_pair_idx[0].shape[0]
        # for i in range(1,len(proposals)):
        #     rel_p = self.find_label(proposals[i].get_field("labels"), rel_pair_idx[i])
        #     edge_idx_ = rel_labels[]
        #     no_zero_indice_ = torch.nonzero(edge_idx_).squeeze(dim=1)
        #     edge_idx_ = edge_idx_.unsqueeze(dim=0).t()
        #     triplet_head_ = rel_p[:,:1]
        #     triplet_tail_ = rel_p[:,1:]
        #     triplet_ = cat((triplet_head_,edge_idx_),dim=1)
        #     triplet_ = cat((triplet_,triplet_tail_),dim=1)
        #     triplet_ = triplet_[no_zero_indice_,:]

        #     triplet = cat((triplet,triplet_),dim=0)

        unique_triplet = torch.unique(triplet,dim=0)
        return unique_triplet
    

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds

    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        for i ,layer in enumerate(self.layers):
            layer_init(layer, xavier=True)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  
        return x
    
    
def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2

@registry.ROI_RELATION_PREDICTOR.register("HetSGG_Predictor")
class HetSGG_Predictor(nn.Module):
    def __init__(self, config, in_channels):
        super(HetSGG_Predictor, self).__init__()
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES # Duplicate
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES # Duplicate
        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS
        self.rel_aware_model_on = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS

        self.rel_aware_loss_eval = None

        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = "predcls" if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else "sgcls"
        else:
            self.mode = "sgdet"
            
        self.obj_recls_logits_update_manner = (
            cfg.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )

        self.n_reltypes = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.NUM_RELATION
        self.n_dim = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.H_DIM
        self.n_ntypes = int(sqrt(self.n_reltypes))
        self.obj2rtype = {(i, j): self.n_ntypes*j+i for j in range(self.n_ntypes) for i in range(self.n_ntypes)}

        self.rel_classifier = build_classifier(self.num_rel_cls, self.num_rel_cls) # Linear Layer
        self.obj_classifier = build_classifier(self.num_obj_cls, self.num_obj_cls)

        self.context_layer = HetSGG(config, in_channels)

        assert in_channels is not None
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)
            self.freq_lambda = nn.Parameter(
                torch.Tensor([1.0]), requires_grad=False
            )  # hurt performance when set learnable


    def init_classifier_weight(self):
        self.obj_classifier.reset_parameters()
        for i in self.n_reltypes:
            self.rel_classifier[i].reset_parameters()


    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
        is_training=True,
    ):
        obj_feats, rel_feats = self.context_layer(roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys, logger, is_training) # GNN
    
        if self.mode == "predcls":
            obj_labels = cat([proposal.get_field("labels") for proposal in inst_proposals], dim=0)
            refined_obj_logits = to_onehot(obj_labels, self.num_obj_cls)
        else:
            refined_obj_logits = self.obj_classifier(obj_feats)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        obj_pred_logits = cat([each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0)

        if self.use_obj_recls_logits:
            boxes_per_cls = cat([proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0)  # comes from post process of box_head
            # here we use the logits refinements by adding
            if self.obj_recls_logits_update_manner == "add":
                obj_pred_logits = refined_obj_logits + obj_pred_logits
            if self.obj_recls_logits_update_manner == "replace":
                obj_pred_logits = refined_obj_logits
            refined_obj_pred_labels = obj_prediction_nms(
                boxes_per_cls, obj_pred_logits, nms_thresh=0.5
            )
            obj_pred_labels = refined_obj_pred_labels
        else:
            obj_pred_labels = cat([each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0)
        
        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = rel_cls_logits + self.freq_bias.index_with_labels(pair_pred.long())

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)
        add_losses = {}

        return obj_pred_logits, rel_cls_logits, add_losses



def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)




@registry.ROI_RELATION_PREDICTOR.register("HetSGGplus_Predictor")
class HetSGGplus_Predictor(nn.Module):
    def __init__(self, config, in_channels):
        super(HetSGGplus_Predictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        assert in_channels is not None
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.input_dim = in_channels
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_HIDDEN_DIM

        self.split_context_model4inst_rel = (
            config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.SPLIT_GRAPH4OBJ_REL
        )
        if self.split_context_model4inst_rel:
            self.obj_context_layer = HetSGGplus_Context(
                config,
                self.input_dim,
                hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
            )
            self.rel_context_layer = HetSGGplus_Context(
                config,
                self.input_dim,
                hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
            )
        else:
            self.context_layer = HetSGGplus_Context(
                config,
                self.input_dim,
                hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
            )

        self.rel_feature_type = config.MODEL.ROI_RELATION_HEAD.EDGE_FEATURES_REPRESENTATION

        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS
        self.obj_recls_logits_update_manner = (
            config.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )
        assert self.obj_recls_logits_update_manner in ["replace", "add"]

        # post classification
        self.rel_classifier = build_classifier(self.hidden_dim, self.num_rel_cls)
        self.obj_classifier = build_classifier(self.hidden_dim, self.num_obj_cls)

        self.rel_aware_model_on = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON

        if self.rel_aware_model_on:
            self.rel_aware_loss_eval = RelAwareLoss(config)

        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        # freq
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)
            self.freq_lambda = nn.Parameter(
                torch.Tensor([1.0]), requires_grad=False
            )  # hurt performance when set learnable

        self.init_classifier_weight()

        # for logging things
        self.forward_time = 0

    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()
        self.obj_classifier.reset_parameters()

    def start_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode()

    def end_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode(False)

    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
        is_training=None
    ):
        """

        :param inst_proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """


        obj_feats, rel_feats, pre_cls_logits, relatedness = self.context_layer(
            roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys, logger
        )

        if relatedness is not None:
            for idx, prop in enumerate(inst_proposals):
                prop.add_field("relness_mat", relatedness[idx])

        if self.mode == "predcls":
            obj_labels = cat(
                [proposal.get_field("labels") for proposal in inst_proposals], dim=0
            )
            refined_obj_logits = to_onehot(obj_labels, self.num_obj_cls)
        else:
            refined_obj_logits = self.obj_classifier(obj_feats)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        obj_pred_logits = cat(
            [each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0
        )

        # using the object results, update the pred label and logits
        if self.use_obj_recls_logits:
            if (self.mode == "sgdet") | (self.mode =="sgcls"):
                boxes_per_cls = cat(
                    [proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0
                )  # comes from post process of box_head
                # here we use the logits refinements by adding
                if self.obj_recls_logits_update_manner == "add":
                    obj_pred_logits = refined_obj_logits + obj_pred_logits
                if self.obj_recls_logits_update_manner == "replace":
                    obj_pred_logits = refined_obj_logits
                refined_obj_pred_labels = obj_prediction_nms(
                    boxes_per_cls, obj_pred_logits, nms_thresh=0.5
                )
                obj_pred_labels = refined_obj_pred_labels
            # else:
            #     _, obj_pred_labels = refined_obj_logits[:, 1:].max(-1)
        else:
            obj_pred_labels = cat(
                [each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0
            )

        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(
                    torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
                )
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = (
                rel_cls_logits
                + self.freq_lambda * self.freq_bias.index_with_labels(pair_pred.long())
            )

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)

        add_losses = {}
        ## pre clser relpn supervision
        if pre_cls_logits is not None and self.training:
            rel_labels = cat(rel_labels, dim=0)
            for iters, each_iter_logit in enumerate(pre_cls_logits):
                if len(squeeze_tensor(torch.nonzero(rel_labels != -1))) == 0:
                    loss_rel_pre_cls = None
                else:
                    loss_rel_pre_cls = self.rel_aware_loss_eval(each_iter_logit, rel_labels)

                add_losses[f"pre_rel_classify_loss_iter-{iters}"] = loss_rel_pre_cls

        return obj_pred_logits, rel_cls_logits, add_losses