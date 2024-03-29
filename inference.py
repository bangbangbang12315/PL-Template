import argparse
import json
import logging
import os
import pickle
import random
from datetime import datetime
from itertools import chain, zip_longest
from os.path import exists, join

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (BertTokenizer, GPT2Config, GPT2LMHeadModel,
                          GPT2TokenizerFast)

from model import MInterface
from data import DInterface
PAD = '[PAD]'
pad_id = 0


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None, type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    # parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
    #                     help='模型参数')
    parser.add_argument('--log_path', default='ref/Selected_Weibo/interact.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--test_data_dir', default='ref/Selected_Weibo/test.txt', type=str, required=False, help='选择测试集')
    parser.add_argument('--inference_path', default='ref/Selected_Weibo/inference.txt', type=str, required=False, help='生成样本地址')
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--config_path', default='pretrained/gpt2-chinese-cluecorpussmall/config.json', type=str)
    # parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=32, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=3, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')

    parser.add_argument('--dataset', default='dialo_dataset', type=str)
    parser.add_argument('--vocab_path', default='pretrained/gpt2-chinese-cluecorpussmall/vocab.txt', type=str)
    # parser.add_argument('--train_data_dir', default='ref/Selected_Weibo/train.txt', type=str)
    # parser.add_argument('--valid_data_dir', default='ref/Selected_Weibo/dev.txt', type=str)
    parser.add_argument('--model_name', default='MSP', type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--pretrained_generator_path', default='pretrained/gpt2-chinese-cluecorpussmall/', type=str)
    parser.add_argument('--pretrained_selector_path', default='pretrained/smn/weibo.SMN.2021-10-06_10:21:55.pt', type=str)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--batch_size', default=1, type=int)

    
    # Model Hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--generator_config', default='pretrained/gpt2-chinese-cluecorpussmall/config.json', type=str)
    parser.add_argument('--max_length', default=512, type=int)
    return parser.parse_args()


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels

def main():
    args = set_args()
    logger = create_logger(args)
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    model_module = MInterface.load_from_checkpoint(checkpoint_path=args.model_path)
    # args.resume_from_checkpoint = args.model_path
    # model_module = MInterface(**vars(args))
    # trainer = Trainer.from_argparse_args(args)

    model = model_module.model
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
        # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    cnt = 0
    data_module = DInterface(**vars(args))
    data_module.setup('test')
    test_dataloader = data_module.test_dataloader()
    if args.gpus != 'cpu':
        device = torch.device('cuda:{}'.format(args.gpus))  
    else:
        device = torch.device('cpu')
    model = model.to(device)
    pass_value = 0
    with open(args.inference_path, 'a+') as ftgt:
        with torch.no_grad():
            for batch_idx, input_ids in enumerate(test_dataloader):
                # visualize(model, tokenizer, input_ids)
                pass_value += 1
                if pass_value <= 1016:
                    continue
                post, resp, ref = input_ids["post"], input_ids["resp"], input_ids["ref"]
                post = post.to(device)
                resp = resp.to(device)
                ref = ref.to(device)
                # input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
                poststr = ''.join(tokenizer.convert_ids_to_tokens(post.squeeze()))
                generated = []
                ref_keep = model.selector.extract_M(post, ref).detach() #[batch_size, turn_num, context_len]
                ref_keep = ref_keep.view(post.size(0), -1)
                ref_str = ''.join(tokenizer.convert_ids_to_tokens(ref_keep.squeeze()))
                print(poststr, ref_str)
                for _ in range(args.max_len):
                    outputs = model.generator(post, ref_keep)
                    logits = outputs.logits
                    next_token_logits = logits[0][-1, :]
                    
                    # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                    for id in set(generated):
                        next_token_logits[id] /= args.repetition_penalty
                    next_token_logits = next_token_logits / args.temperature
                    # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                    next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                    # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                        break
                    generated.append(next_token.item())
                    post = torch.cat((post, next_token.unsqueeze(0)), dim=1)
                text = ''.join(tokenizer.convert_ids_to_tokens(generated))
                ftgt.write(text + '\n')
                if cnt < 100:
                    cnt += 1
                    print('Post: ', poststr)
                    print('Response: ', text)

if __name__ == '__main__':
    main()
