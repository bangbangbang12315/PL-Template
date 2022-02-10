'''
Author: your name
Date: 2022-01-21 16:43:22
LastEditTime: 2022-02-08 16:39:16
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /PL-Template/main.py
'''
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from configParser import args
from data import DInterface
from model import MInterface
from utils import load_model_path_by_args

def load_callbacks():
    callbacks = []
    # callbacks.append(plc.EarlyStopping(
    #     monitor='val_acc',
    #     mode='max',
    #     patience=10,
    #     min_delta=0.001
    # ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=5,
        mode='min',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))
    
    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.resume_from_checkpoint = load_path
        # model = MInterface.load_from_checkpoint(checkpoint_path=load_path)

    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
    args.callbacks = load_callbacks()
    trainer = Trainer.from_argparse_args(args)
    # args.logger = logger
    if args.is_test:    
        trainer.test(model, data_module)
    else:
        trainer.fit(model, data_module)
        
if __name__ == '__main__':
    main(args)
