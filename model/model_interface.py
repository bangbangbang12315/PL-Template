'''
Author: your name
Date: 2022-01-21 16:43:22
LastEditTime: 2022-02-10 22:39:24
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /PL-Template/model/model_interface.py
'''
import importlib
import inspect
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from .utils import calculate_acc

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, input_ids=None):
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        #train generator
        input_ids = batch["input_ids"]
        outputs = self(input_ids)
        loss = outputs.loss
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        outputs = self(input_ids)
        loss = outputs.loss
        logits = outputs.logits
        n_correct, n_word = calculate_acc(logits, input_ids, ignore_index=0)
        self.log('val_acc', n_correct / n_word,
                 on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return None

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)
        if self.hparams.lr_scheduler is None:
            return [optimizer], []
        else:
            t_total = 100000
            scheduler = get_linear_schedule_with_warmup(optimizer, self.hparams.warm_up_steps, t_total)
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'ce':
            self.loss_function = F.cross_entropy
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        Model = getattr(importlib.import_module(
                '.'+name, package=__package__), name)
        self.model = self.instancialize(Model)
        # print(self.pretrained,self.hparams)
        if self.hparams.pretrained:
            if self.hparams.pretrained_generator_path:
                self.model.generator.load_weight(self.hparams.pretrained_generator_path)
            if self.hparams.pretrained_selector_path:
                self.model.selector.load_weight(self.hparams.pretrained_selector_path)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        # class_args = inspect.getargspec(Model.__init__).args[1:]
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
