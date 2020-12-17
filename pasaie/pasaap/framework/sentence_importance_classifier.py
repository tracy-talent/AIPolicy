import os
import datetime
import time
import operator
from collections import defaultdict
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from ...utils.adversarial import FGM, PGD, FreeLB, adversarial_perturbation
from .data_loader import SentenceImportanceDataset, get_train_val_dataloader
from ...metrics import BatchMetric, Mean
from ...losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy


class SentenceImportanceClassifier(nn.Module):
    """model(adaptive) + crf decoder"""

    def __init__(self,
                 model,
                 csv_path,
                 ckpt,
                 logger,
                 tb_logdir,
                 compress_seq=False,
                 neg_classes=[0],
                 batch_size=32,
                 max_epoch=100,
                 lr=1e-3,
                 bert_lr=3e-5,
                 weight_decay=1e-5,
                 early_stopping_step=3,
                 warmup_step=300,
                 max_grad_norm=5.0,
                 dice_alpha=0.6,
                 recall_alpha=0.7,         # used when need to emphasize recall score
                 sampler=None,
                 metric='micro_f1',
                 loss='ce',
                 adv='fgm',
                 opt='adam'):

        super(SentenceImportanceClassifier, self).__init__()
        if 'bert' in model.sequence_encoder.__class__.__name__.lower():
            self.is_bert_encoder = True
        else:
            self.is_bert_encoder = False
        self.max_epoch = max_epoch
        self.early_stopping_step = early_stopping_step
        self.max_grad_norm = max_grad_norm
        self.recall_alpha = recall_alpha
        self.metric = metric

        # Load Data
        self.train_loader, self.eval_loader = get_train_val_dataloader(
            csv_path=csv_path,
            sequence_encoder=model.sequence_encoder,
            batch_size=batch_size,
            sampler=sampler,
            compress_seq=compress_seq
        )
        # ignore_classes for selecting model
        self.neg_classes = neg_classes

        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        if loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif loss == 'pwbce':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.train_loader.dataset.pos_weight, reduction='none')
        elif loss == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif loss == 'wce':
            self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight, reduction='none')
        elif loss == 'focal':
            self.criterion = FocalLoss(gamma=2., reduction='none')
        elif loss == 'dice':
            self.criterion = DiceLoss(alpha=dice_alpha, gamma=0., reduction='none')
        elif loss == 'lsr':
            self.criterion = LabelSmoothingCrossEntropy(eps=0.1, reduction='none')
        else:
            raise ValueError("Invalid loss. Must be 'bce', 'ce' or 'focal' or 'dice' or 'lsr'")
        # Params and optimizer
        self.lr = lr
        self.bert_lr = bert_lr
        if self.is_bert_encoder:
            encoder_params = self.parallel_model.module.sequence_encoder.parameters()
            bert_params_id = list(map(id, encoder_params))
        else:
            encoder_params = []
            bert_params_id = []
        bert_params = list(filter(lambda p: id(p) in bert_params_id, self.parallel_model.parameters()))
        other_params = list(filter(lambda p: id(p) not in bert_params_id, self.parallel_model.parameters()))
        grouped_params = [
            {'params': bert_params, 'lr': bert_lr},
            {'params': other_params, 'lr': lr}
        ]
        if opt == 'sgd':
            self.optimizer = optim.SGD(grouped_params, weight_decay=weight_decay, lr=self.lr)
        elif opt == 'adam':
            self.optimizer = optim.Adam(grouped_params)  # adam weight_decay is not reasonable
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
                    'params': [p for n, p in params if
                               not any(nd in n for nd in no_decay) and id(p) not in bert_params_id],
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
                'early_stopping_best_val': 1e8,
                'epoch_index': 0,
                'train_metrics': [], # [{'loss':0, 'acc':0, 'micro_p':0, 'micro_r':0, 'micro_f1':0, 'alpha_f1':0}]
                'val_metrics': [], # [{'loss':0, 'acc':0, 'micro_p':0, 'micro_r':0, 'micro_f1':0, 'alpha_f1':0}]
                }
    
    
    def update_train_state(self):
        if 'loss' in self.metric:
            cmp_op = operator.gte
        else:
            cmp_op = operator.lte
        if train_state['epoch_index'] == 0:
            self.save_model(self.ckpt)
            train_state['early_stopping_best_val'] = train_state['val_metrics'][-1][self.metric]
            self.logger.info("Best ckpt and saved.")
        elif train_state['epoch_index'] >= 1:
            metric_v2 = train_state['val_metrics'][-2][self.metric]
            metric_v1 = train_state['val_metrics'][-1][self.metric]
            if cmp_op(metric_v1, metric_v2):
                train_state['early_stopping_step'] += 1
            else:
                if not cmp_op(metric_v1, train_state['early_stopping_best_val']):
                    self.save_model(self.ckpt)
                    train_state['early_stopping_best_val'] = metric_v1
                    self.logger.info("Best ckpt and saved.")
                train_state['early_stopping_step'] = 0
            
            train_state['stop_early'] = train_state['early_stopping_step'] >= self.early_stopping_step


    def train_model(self):
        test_best_metric = 0
        best_metric = 0
        global_step = 0
        train_state = self.make_train_state()

        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
            train_state['epoch_index'] = epoch
            batch_metric = BatchMetric(num_classes=max(self.model.num_classes, 2), ignore_classes=self.neg_classes)
            avg_loss = Mean()
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
                if logits.size(-1) == 1:
                    pred = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
                else:
                    pred = logits.argmax(dim=-1)  # (B)
                bs = label.size(0)

                # Optimize
                loss_label = label
                if 'bce' in self.criterion.__class__.__name__.lower():
                    loss_label = label.float().unsqueeze(-1)
                if self.adv is None:
                    loss = self.criterion(logits, loss_label)  # B
                    loss = loss.mean()
                    loss.backward()
                else:
                    loss = adversarial_perturbation(self.adv, self.parallel_model, self.criterion, 3, 0., loss_label, *args)
                loss_label = loss_label.detach().cpu()
                torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.warmup_step > 0:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                # metrics
                batch_metric.update(pred, label)
                avg_loss.update(loss.item() * bs, bs)
                cur_loss = avg_loss.avg
                cur_acc = batch_metric.accuracy().item()
                cur_prec = batch_metric.precision().item()
                cur_rec = batch_metric.recall().item()
                cur_f1 = batch_metric.f1_score().item()

                # log
                global_step += 1
                if global_step % 20 == 0:
                    self.logger.info(f'Training...Epoches: {epoch}, steps: {global_step}, loss: {cur_loss:.4f},'
                                     f' acc: {cur_acc:.4f}, micro_p: {cur_prec:.4f}, micro_r: {cur_rec:.4f},'
                                     f' micro_f1: {cur_f1:.4f}')

                # tensorboard training writer
                if global_step % 5 == 0:
                    self.writer.add_scalar('train loss', cur_loss, global_step=global_step)
                    self.writer.add_scalar('train acc', cur_acc, global_step=global_step)
                    self.writer.add_scalar('train micro precision', cur_prec, global_step=global_step)
                    self.writer.add_scalar('train micro recall', cur_rec, global_step=global_step)
                    self.writer.add_scalar('train micro f1', cur_f1, global_step=global_step)
            alpha_f1 = self.recall_alpha * cur_rec + (1 - self.recall_alpha) * cur_prec
            train_state['train_metrics'].append({'loss': cur_loss, 'acc': cur_acc, 'micro_p': cur_prec, 'micro_r': cur_rec, 'micro_f1': cur_f1, 'alpha_f1': alpha_f1})

            # Val
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.eval_loader)
            self.logger.info('Evaluation result: {}.'.format(result))
            self.logger.info('Metric {} current / best: {} / {}'.format(self.metric, result[self.metric], train_state['early_stopping_best_val']))
            category_result = result.pop('category-p/r/f1')
            train_state['val_metrics'].append(result)
            result['category-p/r/f1'] = category_result
            self.update_train_state(self.metric)
            if not self.warmup_step > 0:
                self.scheduler.step(train_state['val_metrics'][-1][self.metric])
            if train_state['stop_early']:
                break

            # tensorboard val writer
            self.writer.add_scalar('val acc', result['acc'], epoch)
            self.writer.add_scalar('val micro precision', result['micro_p'], epoch)
            self.writer.add_scalar('val micro recall', result['micro_r'], epoch)
            self.writer.add_scalar('val micro f1', result['micro_f1'], epoch)
            print(f"Training time for epoch{epoch}: {batch_metric.epoch_time_interval()}s")

            # test
            if hasattr(self, 'test_loader'):
                result = self.eval_model(self.test_loader)
                self.logger.info('Test result: {}.'.format(result))
                self.logger.info('Metric {} current / best: {} / {}'.format(self.metric, result[self.metric], test_best_metric))
                if result[self.metric] > test_best_metric:
                    self.logger.info('Best test ckpt and saved')
                    self.save_model(self.ckpt[:-9] + '_test' + self.ckpt[-9:])
                    # torch.save({'model': self.model.state_dict()}, self.ckpt[:-9] + '_test' + self.ckpt[-9:])
                    test_best_metric = result[self.metric]

        self.logger.info("Best %s on val set: %f" % (self.metric, train_state['early_stopping_best_val']))
        if hasattr(self, 'test_loader'):
            self.logger.info("Best %s on test set: %f" % (self.metric, test_best_metric))


    def eval_model(self, eval_loader):
        self.eval()

        batch_metric = BatchMetric(num_classes=max(self.model.num_classes, 2), ignore_classes=self.neg_classes)
        avg_loss = Mean()
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
                bs = label.size(0)

                if logits.size(-1) == 1:
                    pred = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
                else:
                    pred = logits.argmax(dim=-1)  # (B)
                batch_metric.update(pred, label)
                if 'bce' in self.criterion.__class__.__name__.lower():
                    label = label.float().unsqueeze(-1)
                loss = self.criterion(logits, label).mean().item()
                avg_loss.update(loss * bs, bs)
                # log
                if (ith + 1) % 20 == 0:
                    self.logger.info(f'Evaluation...steps: {ith + 1} finished')

        loss = avg_loss.avg
        acc = batch_metric.accuracy().item()
        micro_prec = batch_metric.precision().item()
        micro_rec = batch_metric.recall().item()
        micro_f1 = batch_metric.f1_score().item()

        alpha_f1 = self.recall_alpha * micro_rec + (1 - self.recall_alpha) * micro_prec
        cate_prec = batch_metric.precision('none')
        cate_rec = batch_metric.recall('none')
        cate_f1 = batch_metric.f1_score('none')
        category_result = {k: v for k, v in enumerate(zip(cate_prec, cate_rec, cate_f1))}
        result = {'loss': loss, 'acc': acc, 'micro_p': micro_prec, 'micro_r': micro_rec, 'micro_f1': micro_f1,
                  'alpha_f1': alpha_f1, 'category-p/r/f1': category_result}
        return result

    def load_model(self, ckpt):
        self.model = torch.load(ckpt)
    
    def save_model(self, ckpt):
        torch.save(self.model, ckpt)
