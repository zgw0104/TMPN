import torch
import torch.nn as nn
from transformers import LxmertModel,LxmertForPreTraining, LxmertTokenizer
from .utils_relation import layer_init
import json
import h5py

class GlobalPromptLxmert(nn.Module):
    def __init__(self, pretrained_model_name='/home/yj/zgw/lxmert-base-uncased', prompt_length=3):
        super().__init__()
        self.lxmert = LxmertForPreTraining.from_pretrained(pretrained_model_name)
        self.lxmert2 = LxmertModel.from_pretrained(pretrained_model_name)

        self.tokenizer = LxmertTokenizer.from_pretrained(pretrained_model_name)
        self.prompt_length = prompt_length
        embed_dim = self.lxmert.config.hidden_size #768
        
        # 定义全局共享提示参数
        self.global_prompts = nn.Parameter(torch.randn(prompt_length, embed_dim))
        with open('/home/yj/zgw/het2/Datasets/VG/idx_to_predicate.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.entity_word_list = list(data['idx_to_label'].values())
        self.predicate_word_list = list(data['idx_to_predicate'].values())
        self.predicate_word_list = [str(predicate) for predicate in self.predicate_word_list]


        self.lxmert_entity_idList = self.tokenizer.convert_tokens_to_ids(self.entity_word_list)
        self.lxmert_predicate_idList = self.tokenizer.convert_tokens_to_ids(self.predicate_word_list)

        self.predicate_idx = {word: idx for idx, word in enumerate(self.lxmert_predicate_idList)}

        struct_proto = '/home/yj/zgw/het2/proto/struct_p.h5'
        f2 = h5py.File(struct_proto,'r')
        self.struct_proto = torch.from_numpy(f2['struct_proto'][:])
        # self.struct_proto = torch.from_numpy(self.struct_proto)
        self.graph_prompt = nn.Parameter(self.struct_proto)
        
        self.dev = nn.Linear(2048,768)
        layer_init(self.dev,xavier=True)
        # 冻结预训练参数，仅训练提示
        for param in self.lxmert.parameters():
            param.requires_grad = False

    def _get_token_positions(self, input_ids):
        """获取 a 和 b 在输入序列中的位置索引"""
        # 将 input_ids 转换为 token
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        
        # 假设模板为 <a, [mask], b>，找到 a 和 b 的位置
        a_pos = None
        b_pos = None
        for idx, token in enumerate(tokens):
            if token == 'a':  # 替换为实际 token
                a_pos = idx
            elif token == 'b':  # 替换为实际 token
                b_pos = idx
        return a_pos, b_pos

    def forward(self, input_ids, visual_feats, visual_pos, token_type_ids=None, attention_mask=None):
        # 获取文本嵌入
        text_embeds = self.lxmert.lxmert.embeddings.word_embeddings(input_ids)

        
        # 在文本输入前拼接全局提示
        batch_size = text_embeds.size(0)
        
        
        
        
        # 前向传播
        outputs = self.lxmert(
            inputs_embeds=text_embeds,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        mask_positions = torch.where(input_ids == self.tokenizer.mask_token_id)[1]
        batch_logits = outputs.prediction_logits[torch.arange(text_embeds.shape[0]), mask_positions]
        batch_logits[:, ~torch.isin(torch.arange(batch_logits.shape[-1]), torch.tensor(self.lxmert_predicate_idList))] = -float('inf')
        predicted_ids = torch.argmax(batch_logits, dim=-1)
        indice = torch.tensor([self.predicate_idx.get(value.item(), -1) for value in predicted_ids])

        graph_prompt = self.dev(self.graph_prompt[indice,:]).unsqueeze(1)

        #prompt = [cls,sub,mask,obj,end]
        prompts = self.global_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        start = text_embeds[:,0,:].unsqueeze(1) #[cls]
        text_embeds_tmp = torch.cat([start,prompts], dim=1) #[cls,p1,p2,p3]
        text_embeds_tmp = torch.cat([text_embeds_tmp,text_embeds[:,1,:].unsqueeze(1)], dim=1)#[cls,p1,p2,p3,sub]
        text_embeds_tmp = torch.cat([text_embeds_tmp,graph_prompt], dim=1)#[cls,p1,p2,p3,sub,g1]
        text_embeds = torch.cat([text_embeds_tmp, text_embeds[:,1:,:]], dim=1) #[cls,p1,p2,p3,sub,g1,mask,obj,end]

        # 调整attention_mask以包含提示
        if attention_mask is not None:
            prompt_mask = torch.ones((batch_size, self.prompt_length+2), device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.lxmert2(
            inputs_embeds=text_embeds,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            attention_mask=attention_mask,
            output_hidden_states=True
        )


        last_hidden_state = outputs.language_hidden_states[-1]
        # # 确定 sub 和 obj 的位置
        # a_pos, b_pos = self._get_token_positions(input_ids)
        
        # # 提取嵌入
        sub_embeddings = last_hidden_state[:, 4, :]  # [batch_size, hidden_size]
        obj_embeddings = last_hidden_state[:, 7, :]  # [batch_size, hidden_size]
        rel_embeddings = last_hidden_state[:, 6, :]

        sub_vis_embedding = outputs.vision_hidden_states[-1][:,0,:]
        obj_vis_embedding = outputs.vision_hidden_states[-1][:,1,:]

        cls_embedding = outputs.pooled_output


        return sub_embeddings, obj_embeddings, rel_embeddings,indice, sub_vis_embedding, obj_vis_embedding,cls_embedding