'''
Author: your name
Date: 2022-01-21 16:43:22
LastEditTime: 2022-02-10 22:41:06
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /PL-Template/data/dialo_dataset.py
'''
from collections import defaultdict
import os
from io import SEEK_CUR
from turtle import pos
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer

class DialoDataset(Dataset):
    def __init__(self, train_data_dir=None, valid_data_dir=None, test_data_dir=None, train=False, config_path=None, max_length=512):
        self.tok = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=config_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
        self.max_length = max_length
        self.train = train
        self.train_data_dir = train_data_dir
        self.valid_data_dir = valid_data_dir
        self.test_data_dir = test_data_dir
        if self.train:
            self.data = self.load_data(train_data_dir)
        else:
            if valid_data_dir != None:
                self.data = self.load_data(valid_data_dir)
            else:
                self.data = self.load_data(test_data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        tokenized_line = defaultdict(list)
        for k, v in line.items():
            if k == 'input_ids':
                max_len = self.max_length
            else:
                max_len = self.max_length // 2
            tokenized_line[k] = self.tok.encode(
                                    v,
                                    max_length=max_len,
                                    truncation=True,
                                    return_tensors="pt"
                                ).squeeze()
        return tokenized_line
    
    def load_data(self, data_dir):
        '''
        line: post###resp
        '''
        data = []
        with open(data_dir, 'r') as fsrc:
            for sub in tqdm(fsrc, desc='Load Dataset'):
                post, resp = sub.strip().split('\t')
                sub_dict = {'post': '[CLS]' + post,
                            'resp': resp,
                            'input_ids': post + '[SEP]' + resp + '[EOS]'}
                if len(post) == 0 or len(resp) == 0:
                    continue
                data.append(sub_dict)
        return data
