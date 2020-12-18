from ...utils.adversarial import adversarial_perturbation
from .data_loader import SentenceImportanceDataset, get_train_val_dataloader
from ...metrics import BatchMetric, Mean
from .base_framework import BaseFramework

import os
from collections import defaultdict

import torch
from torch import nn


class SentenceImportanceClassifier(BaseFramework):
    """sentence classifier"""

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

        # Load Data
        self.train_loader, self.eval_loader = get_train_val_dataloader(
            csv_path=csv_path,
            sequence_encoder=model.sequence_encoder,
            batch_size=batch_size,
            sampler=sampler,
            compress_seq=compress_seq
        )

        # initialize base class
        super(SentenceImportanceClassifier, self).__init__(
            model=model,
            ckpt=ckpt,
            logger=logger,
            tb_logdir=tb_logdir,
            batch_size=batch_size,
            max_epoch=max_epoch,
            lr=lr,
            bert_lr=bert_lr,
            weight_decay=weight_decay,
            early_stopping_step=early_stopping_step,
            warmup_step=warmup_step,
            max_grad_norm=max_grad_norm,
            metric=metric,
            opt=opt,
            loss=loss,
            adv=adv,
            dice_alpha=dice_alpha,
            loss_weight=self.train_loader.dataset.weight,
            pos_weight=self.train_loader.dataset.pos_weight
        )

        self.recall_alpha = recall_alpha


    def train_model(self):
        test_best_metric = 1e8 if 'loss' in self.metric else 0
        best_metric = test_best_metric
        global_step = 0
        train_state = self.make_train_state()

        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
            train_state['epoch_index'] = epoch
            avg_loss = Mean()
            batch_metric = BatchMetric(num_classes=max(self.model.num_classes, 2), ignore_classes=self.neg_classes)
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
                cur_acc = batch_metric.accuracy()
                cur_prec = batch_metric.precision()
                cur_rec = batch_metric.recall()
                cur_f1 = batch_metric.f1_score()

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
            print(f"Training time for epoch{epoch}: {batch_metric.epoch_time_interval()}s")

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
                    # torch.save({'model': self.model.state_dict()}, self.ckpt[:-9] + '_test' + self.ckpt[-9:])
                    test_best_metric = result[self.metric]

        self.logger.info("Best %s on val set: %f" % (self.metric, train_state['early_stopping_best_val']))
        if hasattr(self, 'test_loader'):
            self.logger.info("Best %s on test set: %f" % (self.metric, test_best_metric))


    def eval_model(self, eval_loader):
        self.eval()
        avg_loss = Mean()
        batch_metric = BatchMetric(num_classes=max(self.model.num_classes, 2), ignore_classes=self.neg_classes)

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
                # metrics
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
        acc = batch_metric.accuracy()
        micro_prec = batch_metric.precision()
        micro_rec = batch_metric.recall()
        micro_f1 = batch_metric.f1_score()

        alpha_f1 = self.recall_alpha * micro_rec + (1 - self.recall_alpha) * micro_prec
        cate_prec = batch_metric.precision('none')
        cate_rec = batch_metric.recall('none')
        cate_f1 = batch_metric.f1_score('none')
        category_result = {k: v for k, v in enumerate(zip(cate_prec, cate_rec, cate_f1))}
        result = {'loss': loss, 'acc': acc, 'micro_p': micro_prec, 'micro_r': micro_rec, 'micro_f1': micro_f1,
                  'alpha_f1': alpha_f1, 'category-p/r/f1': category_result}
        return result
