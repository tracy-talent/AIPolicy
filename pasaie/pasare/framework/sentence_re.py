"""
 Author: liujian
 Date: 2020-12-18 12:47:45
 Last Modified by: liujian
 Last Modified time: 2020-12-18 12:47:45
"""

from ...metrics import Mean, BatchMetric
from ...losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy
from ...utils.adversarial import FGM, PGD, FreeLB, adversarial_perturbation
from .data_loader import SentenceRELoader, SentenceWithDSPRELoader

import os
import datetime
import operator
from collections import defaultdict

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter


class SentenceRE(nn.Module):

    def __init__(self,
                 model,
                 train_path,
                 val_path,
                 test_path,
                 ckpt,
                 logger,
                 tb_logdir,
                 neg_classes=[0],
                 compress_seq=True,
                 batch_size=32,
                 max_epoch=100,
                 lr=1e-3,
                 bert_lr=2e-5,
                 weight_decay=1e-5,
                 early_stopping_step=3,
                 warmup_step=300,
                 max_grad_norm=5.0,
                 dice_alpha=0.6,
                 metric='micro_f1',
                 adv='fgm',
                 loss='ce',
                 opt='sgd',
                 sampler=None):

        super().__init__()
        if 'bert' in model.sentence_encoder.__class__.__name__.lower():
            self.is_bert_encoder = True
        else:
            self.is_bert_encoder = False
        self.max_grad_norm = max_grad_norm
        self.max_epoch = max_epoch
        self.metric = metric
        self.early_stopping_step = early_stopping_step

        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle=True,
                compress_seq=compress_seq,
                sampler=sampler
            )

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                compress_seq=compress_seq,
                shuffle=False
            )

        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                compress_seq=compress_seq,
                shuffle=False
            )
        # ignore_classes for selecting model
        self.neg_classes = neg_classes

        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        if loss == 'ce':
            # nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif loss == 'focal':
            self.criterion = FocalLoss(gamma=2., reduction='none')
        elif loss == 'dice':
            self.criterion = DiceLoss(alpha=dice_alpha, gamma=0., reduction='none')
        elif loss == 'lsr':
            self.criterion = LabelSmoothingCrossEntropy(eps=0.1, reduction='none')
        else:
            raise ValueError("Invalid loss. Must be 'ce' or 'focal' or 'dice' or 'lsr'")
        # Params and optimizer
        self.lr = lr
        self.bert_lr = bert_lr
        if self.is_bert_encoder:
            encoder_params = self.parallel_model.module.sentence_encoder.parameters()
            bert_params_id = list(map(id, encoder_params))
        else:
            encoder_params = []
            bert_params_id = []
        bert_params = list(filter(lambda p: id(p) in bert_params_id, self.parallel_model.parameters()))
        other_params = list(filter(lambda p: id(p) not in bert_params_id, self.parallel_model.parameters()))
        grouped_params = [
            {'params': bert_params, 'lr': bert_lr},
            {'params': other_params, 'lr':lr}
        ]
        if opt == 'sgd':
            self.optimizer = optim.SGD(grouped_params, weight_decay=weight_decay) 
        elif opt == 'adam':
            self.optimizer = optim.Adam(grouped_params) # adam weight_decay is not reasonable
            # self.optimizer = optim.Adam(grouped_params, lr, weight_decay=weight_decay) # adam weight_decay is not reasonable
        elif opt == 'adamw':  # Optimizer for BERT
            from transformers import AdamW
            params = list(self.parallel_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            adamw_grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay) and id(p) in bert_params_id], 
                    'weight_decay': weight_decay,
                    'lr': bert_lr,
                },
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay) and id(p) not in bert_params_id], 
                    'weight_decay': weight_decay,
                    'lr': lr,
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay) and id(p) in bert_params_id], 
                    'weight_decay': 0.0,
                    'lr': bert_lr,
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay) and id(p) not in bert_params_id], 
                    'weight_decay': 0.0,
                    'lr': lr,
                }
            ]
            self.optimizer = AdamW(adamw_grouped_params, correct_bias=True)  # original: correct_bias=False
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        self.warmup_step = warmup_step
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                mode='min' if 'loss' in self.metric else 'max', factor=0.8, 
                                                                patience=1, min_lr=5e-6) # mode='min' for loss, 'max' for acc/p/r/f1
            # self.scheduler = None
        # Adversarial
        if adv == 'fgm':
            self.adv = FGM(model=self.parallel_model, emb_name='word_embeddings', epsilon=1.0)
        elif adv == 'pgd':
            self.adv = PGD(model=self.parallel_model, emb_name='word_embeddings', epsilon=1., alpha=0.3)
        elif adv == 'flb':
            self.adv = FreeLB(model=self.parallel_model, emb_name='word_embeddings', epsilon=1., alpha=0.3)
        else:
            self.adv = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt
        # logger
        self.logger = logger
        # tensorboard writer
        self.writer = SummaryWriter(tb_logdir, filename_suffix=datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
        

    def make_train_state(self):
        return {'stop_early': False,
                'early_stopping_step': 0,
                'early_stopping_best_val': 1e8 if 'loss' in self.metric else 0,
                'epoch_index': 0,
                'train_metrics': [], # [{'loss':0, 'acc':0, 'micro_p':0, 'micro_r':0, 'micro_f1':0}]
                'val_metrics': [], # [{'loss':0, 'acc':0, 'micro_p':0, 'micro_r':0, 'micro_f1':0}]
                }
    
    
    def update_train_state(self, train_state):
        if 'loss' in self.metric:
            cmp_op = operator.lt
        else:
            cmp_op = operator.gt
        if train_state['epoch_index'] == 0:
            self.save_model(self.ckpt)
            train_state['early_stopping_best_val'] = train_state['val_metrics'][-1][self.metric]
            self.logger.info("Best ckpt and saved.")
        elif train_state['epoch_index'] >= 1:
            metric_v2 = train_state['val_metrics'][-2][self.metric]
            metric_v1 = train_state['val_metrics'][-1][self.metric]
            if not cmp_op(metric_v1, metric_v2):
                train_state['early_stopping_step'] += 1
            else:
                if cmp_op(metric_v1, train_state['early_stopping_best_val']):
                    self.save_model(self.ckpt)
                    train_state['early_stopping_best_val'] = metric_v1
                    self.logger.info("Best ckpt and saved.")
                train_state['early_stopping_step'] = 0
            
            train_state['stop_early'] = train_state['early_stopping_step'] >= self.early_stopping_step


    def train_model(self):
        test_best_metric = 1e8 if 'loss' in self.metric else 0
        global_step = test_best_metric
        train_state = self.make_train_state()
        negid = -1
        for rel_name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
            if rel_name in self.model.rel2id:
                negid = self.model.rel2id[rel_name]
                break
        if negid == -1:
            raise Exception("negtive tag not in ['NA', 'na', 'no_relation', 'Other', 'Others']")

        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
            train_state['epoch_index'] = epoch
            avg_loss = Mean()
            batch_metric = BatchMetric(num_classes=max(len(self.model.rel2id), 2), ignore_classes=[negid])
            for ith, data in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                logits = self.parallel_model(*args)
                pred = logits.argmax(dim=-1)  # (B)
                bs = label.size(0)

                # Optimize
                if self.adv is None:
                    loss = self.criterion(logits, label)  # B
                    loss = loss.mean()
                    loss.backward()
                else:
                    loss = adversarial_perturbation(self.adv, self.parallel_model, self.criterion, 3, 0., label, *args)
                # torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.warmup_step > 0:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                # metrics
                batch_metric.update(pred, label)
                avg_loss.update(loss.item() * bs, bs)
                cur_loss = avg_loss.avg
                cur_acc = batch_metric.accuracy()
                cur_prec = batch_metric.precision()
                cur_rec = batch_metric.recall()
                cur_f1 = batch_metric.f1_score()
                
                # log
                global_step += 1
                if global_step % 20 == 0:
                    self.logger.info(f'Training...Epoches: {epoch}, steps: {global_step}, loss: {cur_loss:.4f}, acc: {cur_acc:.4f}, micro_p: {cur_prec:.4f}, micro_r: {cur_rec:.4f}, micro_f1: {cur_f1:.4f}')

                # tensorboard training writer
                if global_step % 5 == 0:
                    self.writer.add_scalar('train loss', cur_loss, global_step=global_step)
                    self.writer.add_scalar('train acc', cur_acc, global_step=global_step)
                    self.writer.add_scalar('train micro precision', cur_prec, global_step=global_step)
                    self.writer.add_scalar('train micro recall', cur_rec, global_step=global_step)
                    self.writer.add_scalar('train micro f1', cur_f1, global_step=global_step)
            train_state['train_metrics'].append({'loss': cur_loss, 'acc': cur_acc, 'micro_p': cur_prec, 'micro_r': cur_rec, 'micro_f1': cur_f1})

            # Val 
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            self.logger.info('Evaluation result: {}.'.format(result))
            self.logger.info('Metric {} current / best: {} / {}'.format(self.metric, result[self.metric], train_state['early_stopping_best_val']))
            category_result = result.pop('category-p/r/f1')
            train_state['val_metrics'].append(result)
            result['category-p/r/f1'] = category_result
            self.update_train_state(train_state)
            if not self.warmup_step > 0:
                self.scheduler.step(train_state['val_metrics'][-1][self.metric])
            if train_state['stop_early']:
                break

            # tensorboard val writer
            self.writer.add_scalar('val loss', result['loss'], epoch)
            self.writer.add_scalar('val acc', result['acc'], epoch)
            self.writer.add_scalar('val micro precision', result['micro_p'], epoch)
            self.writer.add_scalar('val micro recall', result['micro_r'], epoch)
            self.writer.add_scalar('val micro f1', result['micro_f1'], epoch)

            # test
            if hasattr(self, 'test_loader'):
                self.logger.info("=== Epoch %d test ===" % epoch)
                result = self.eval_model(self.test_loader)
                self.logger.info('Test result: {}.'.format(result))
                self.logger.info('Metric {} current / best: {} / {}'.format(self.metric, result[self.metric], test_best_metric))
                if 'loss' in self.metric:
                    cmp_op = operator.lt
                else:
                    cmp_op = operator.gt
                if cmp_op(result[self.metric], test_best_metric):
                    self.logger.info('Best test ckpt and saved')
                    self.save_model(self.ckpt[:-9] + '_test' + self.ckpt[-9:])
                    test_best_metric = result[self.metric]

        self.logger.info("Best %s on val set: %f" % (self.metric, train_state['early_stopping_best_val']))
        if hasattr(self, 'test_loader'):
            self.logger.info("Best %s on test set: %f" % (self.metric, test_best_metric))


    def eval_model(self, eval_loader):
        self.eval()
        for rel_name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
            if rel_name in self.model.rel2id:
                negid = self.model.rel2id[rel_name]
                break
        if negid == -1:
            raise Exception("negtive tag not in ['NA', 'na', 'no_relation', 'Other', 'Others']")
        avg_loss = Mean()
        batch_metric = BatchMetric(num_classes=max(len(self.model.rel2id), 2), ignore_classes=[negid])

        with torch.no_grad():
            for ith, data in enumerate(eval_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                logits = self.parallel_model(*args)
                pred = logits.argmax(dim=-1)  # (B)
                bs = label.size(0)
                # metrics
                batch_metric.update(pred, label)
                loss = self.criterion(logits, label).mean().item()
                avg_loss.update(loss * bs, bs)
                # log
                if (ith + 1) % 20 == 0:
                    self.logger.info(f'Evaluation...steps: {ith + 1} finished')

        loss = avg_loss.avg
        acc = batch_metric.accuracy()
        micro_prec = batch_metric.precision()
        micro_rec = batch_metric.recall()
        micro_f1 = batch_metric.f1_score()

        cate_prec = batch_metric.precision('none')
        cate_rec = batch_metric.recall('none')
        cate_f1 = batch_metric.f1_score('none')
        category_result = {k: v for k, v in enumerate(zip(cate_prec, cate_rec, cate_f1))}
        result = {'loss': loss, 'acc': acc, 'micro_p': micro_prec, 'micro_r': micro_rec, 'micro_f1': micro_f1, 
                    'category-p/r/f1': category_result}
        return result


    def load_model(self, ckpt):
        state_dict = torch.load(ckpt)
        self.model.load_state_dict(state_dict['model'])


    def save_model(self, ckpt):
        state_dict = {'model': self.model.state_dict()}
        torch.save(state_dict, ckpt)



class SentenceWithDSPRE(SentenceRE):

    def __init__(self,
                 model,
                 train_path,
                 val_path,
                 test_path,
                 ckpt,
                 logger,
                 tb_logdir,
                 neg_classes=[0],
                 compress_seq=True,
                 max_dsp_path_length=-1,
                 dsp_tool='ddp',
                 batch_size=32,
                 max_epoch=100,
                 lr=1e-3,
                 bert_lr=2e-5,
                 weight_decay=1e-2,
                 early_stopping_step=3,
                 warmup_step=300,
                 max_grad_norm=5.0,
                 dice_alpha=0.6,
                 metric='micro_f1',
                 adv='fgm',
                 loss='ce',
                 opt='sgd',
                 sampler=None):

        super(SentenceWithDSPRE, self).__init__(
            model=model,
            train_path=None,
            val_path=None,
            test_path=None,
            ckpt=ckpt,
            logger=logger,
            tb_logdir=tb_logdir,
            neg_classes=neg_classes,
            compress_seq=compress_seq,
            batch_size=batch_size,
            max_epoch=max_epoch,
            lr=lr,
            bert_lr=bert_lr,
            weight_decay=weight_decay,
            early_stopping_step=early_stopping_step,
            warmup_step=warmup_step,
            max_grad_norm=max_grad_norm,
            dice_alpha=dice_alpha,
            metric=metric,
            adv=adv,
            loss=loss,
            opt=opt,
            sampler=sampler
        )
        # Load data
        if train_path != None:
            self.train_loader = SentenceWithDSPRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle=True,
                drop_last=False,
                compress_seq=compress_seq,
                max_dsp_path_length=max_dsp_path_length,
                dsp_tool=dsp_tool,
                is_bert_encoder=self.is_bert_encoder,
                sampler=sampler,
                num_workers=0 if max_dsp_path_length < 0 else 8
            )

        if val_path != None:
            self.val_loader = SentenceWithDSPRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                compress_seq=compress_seq,
                max_dsp_path_length=max_dsp_path_length,
                dsp_tool=dsp_tool,
                is_bert_encoder=self.is_bert_encoder,
                shuffle=False,
                num_workers=0 if max_dsp_path_length < 0 else 8
            )

        if test_path != None:
            self.test_loader = SentenceWithDSPRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                compress_seq=compress_seq,
                max_dsp_path_length=max_dsp_path_length,
                dsp_tool=dsp_tool,
                is_bert_encoder=self.is_bert_encoder,
                shuffle=False,
                num_workers=0 if max_dsp_path_length < 0 else 8
            )
