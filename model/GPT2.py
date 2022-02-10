'''
Author: your name
Date: 2022-01-12 16:05:46
LastEditTime: 2022-02-10 22:37:23
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /PL-Template/model/GPT2.py
'''
from transformers import GPT2Config, GPT2LMHeadModel
import torch
import os
import torch.nn as nn
class GPT2(nn.Module):
    def __init__(self, config_path=None, model_name=None) -> None:
        super(GPT2, self).__init__()
        self.model_name = model_name
        self.config = GPT2Config.from_json_file(os.path.join(config_path,'config.json'))
        self.model = GPT2LMHeadModel(config=self.config)
        self.model.resize_token_embeddings(self.config.vocab_size + 3)
    
    def forward(self, input_ids, ref=None, output_attentions=False):
        labels = input_ids
        if ref != None:
            input_ids = torch.cat([ref, input_ids], dim=-1)
            labels = torch.cat([torch.ones(ref.shape).type_as(ref)*-100, labels], dim=-1)
        labels = torch.where(labels==0, -100, labels)
        # attention_mask = attention_mask
        r = self.model(
            input_ids=input_ids,
            labels=labels,
            output_attentions=output_attentions,
            return_dict=True,
        )
        return r
    
    def load_weight(self, pretrained_path):
        print('Loading Parameters...')
        self.model = self.model.from_pretrained(pretrained_path)


