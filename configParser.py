'''
Author: your name
Date: 2022-02-07 17:12:32
LastEditTime: 2022-02-10 22:41:41
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /PL-Template/configParser.py
'''
from argparse import ArgumentParser

parser = ArgumentParser()
# Basic Training Control
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--gpus', default='1', type=str, required=False, help="设置使用哪些显卡，用逗号分割")
parser.add_argument('--seed', default=1104, type=int)
parser.add_argument('--min_epochs', default=5, type=int)
parser.add_argument('--max_epochs', default=100, type=int)
parser.add_argument('--val_check_interval', default=100000, type=int)
parser.add_argument('--default_root_dir', default='checkpoints', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--distributed_backend', default='dp', type=str)
parser.add_argument('--fast_dev_run', action='store_true')
parser.add_argument('--is_test', action='store_true')
parser.add_argument('--find_unused_parameters', action='store_true')
# LR Scheduler
parser.add_argument('--lr_scheduler', choices=['step', 'cosine','warmup'], type=str)
parser.add_argument('--optimizer', choices=['Adam', 'AdamW'], type=str)
parser.add_argument('--lr_decay_steps', default=20, type=int)
parser.add_argument('--lr_decay_rate', default=0.5, type=float)
parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)
parser.add_argument('--warm_up_steps', default=4000, type=int, required=False, help="warm up步数")
parser.add_argument('--accumulate_grad_batches', default=1, type=int, required=False)
parser.add_argument('--precision', default=32, type=int, required=False)
parser.add_argument('--gradient_clip_val', default=0.5, type=float)

# Restart Control
parser.add_argument('--load_best', action='store_true')
parser.add_argument('--load_dir', default='checkpoints', type=str)
parser.add_argument('--load_ver', default='version', type=str)
parser.add_argument('--load_v_num', default=3, type=int)

# Training Info
parser.add_argument('--dataset', default='dialo_dataset', type=str)
parser.add_argument('--vocab_path', default='pretrained/gpt2-chinese-cluecorpussmall/vocab.txt', type=str)
parser.add_argument('--train_data_dir', default='ref/train.txt', type=str)
parser.add_argument('--valid_data_dir', default='ref/dev.txt', type=str)
parser.add_argument('--test_data_dir', default='ref/test.txt', type=str)
parser.add_argument('--model_name', default='GPT2', type=str)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--pretrained_generator_path', default=None, type=str)
parser.add_argument('--pretrained_selector_path', default=None, type=str)
parser.add_argument('--word_embeddings', default=None, type=str)

parser.add_argument('--loss', default='ce', type=str)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--no_augment', action='store_true')
parser.add_argument('--log_dir', default='lightning_logs', type=str)

# Model Hyperparameters
parser.add_argument('--config_path', default='pretrained/gpt2-chinese-cluecorpussmall/config.json', type=str)
parser.add_argument('--max_length', default=512, type=int)

args = parser.parse_args()