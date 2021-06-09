"""
 Author: liujian
 Date: 2020-12-05 20:18:03
 Last Modified by: liujian
 Last Modified time: 2020-12-05 20:18:03
"""

from ...losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy, AutomaticWeightedLoss
from ...metrics import Mean, micro_p_r_f1_score
from ...utils.adversarial import FGM, PGD, FreeLB
from ...utils.adversarial import adversarial_perturbation_mrc_span_mtl
from ...utils.entity_extract import extract_kvpairs_by_start_end
from .data_loader import MRCSpanMultiNERDataLoader
from .base_framework import BaseFramework

import os
import operator
import datetime
from collections import defaultdict

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup


class MRC_Span_MTL(nn.Module):
    """train multi task for span_start and span_end"""
    
    def __init__(self, 
                model, 
                train_path, 
                val_path, 
                test_path, 
                query_path,
                ckpt, 
                logger, 
                tb_logdir, 
                compress_seq=True,
                tagscheme='bmoes', 
                batch_size=32, 
                max_epoch=100, 
                lr=1e-3,
                bert_lr=3e-5,
                weight_decay=1e-2,
                early_stopping_step=3,
                warmup_step=300,
                max_grad_norm=5.0,
                metric='micro_f1',
                opt='adam',
                adv='fgm',
                loss='dice',
                add_span_loss=False,
                mtl_autoweighted_loss=True,
                dice_alpha=0.6,
                sampler=None):

        super(MRC_Span_MTL, self).__init__()
        if 'bert' in model.sequence_encoder.__class__.__name__.lower():
            self.is_bert_encoder = True
        else:
            self.is_bert_encoder = False
        self.max_epoch = max_epoch
        self.metric = metric
        self.tagscheme = tagscheme
        self.max_grad_norm = max_grad_norm
        self.early_stopping_step = early_stopping_step
        
        # Load Data
        if train_path != None:
            self.train_loader = MRCSpanMultiNERDataLoader(
                data_path=train_path,
                query_path=query_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=True,
                compress_seq=compress_seq,
                sampler=sampler
            )

        if val_path != None:
            self.val_loader = MRCSpanMultiNERDataLoader(
                data_path=val_path,
                query_path=query_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq
            )
        
        if test_path != None:
            self.test_loader = MRCSpanMultiNERDataLoader(
                data_path=test_path,
                query_path=query_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq
            )

        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(model)
        # Criterion
        if loss == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif loss == 'wce':
            self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight if hasattr(self, 'train_loader') else None, reduction='none')
        elif loss == 'focal':
            self.criterion = FocalLoss(gamma=2., reduction='none')
        elif loss == 'dice':
            self.criterion = DiceLoss(alpha=dice_alpha, gamma=0., reduction='none')
        elif loss == 'lsr':
            self.criterion = LabelSmoothingCrossEntropy(eps=0.1, reduction='none')
        else:
            raise ValueError("Invalid loss. Must be 'ce' or 'focal' or 'dice' or 'lsr'")
        self.span_bce = None
        if add_span_loss:
            self.span_bce = nn.BCEWithLogitsLoss()
        # Automatic weighted loss for mtl(submodule must after torch.nn.Module initialize)
        self.autoweighted_loss = None
        if mtl_autoweighted_loss:
            self.autoweighted_loss = AutomaticWeightedLoss(3 if add_span_loss else 2)
        # Params and optimizer
        self.lr = lr
        self.bert_lr = bert_lr
        if self.is_bert_encoder:
            encoder_params = self.parallel_model.module.sequence_encoder.parameters()
            bert_params_id = list(map(id, encoder_params))
        else:
            encoder_params = []
            bert_params_id = []
        bert_params = list(filter(lambda p: id(p) in bert_params_id, self.parameters()))
        other_params = list(filter(lambda p: id(p) not in bert_params_id, self.parameters()))
        grouped_params = [
            {'params': bert_params, 'lr':bert_lr},
            {'params': other_params, 'lr':lr}
        ]
        if opt == 'sgd':
            self.optimizer = optim.SGD(grouped_params, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(grouped_params) # adam weight_decay is not reasonable
        elif opt == 'adamw': # Optimizer for BERT
            params = list(self.named_parameters())
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
            self.optimizer = AdamW(adamw_grouped_params, correct_bias=True) # original: correct_bias=False
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        self.warmup_step = warmup_step
        if warmup_step > 0:
            training_steps = len(self.train_loader) // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=training_steps)
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
                'train_metrics': [], # exp: [{'loss':0, 'acc':0, 'micro_p':0, 'micro_r':0, 'micro_f1':0}]
                'val_metrics': [], # exp: [{'loss':0, 'acc':0, 'micro_p':0, 'micro_r':0, 'micro_f1':0}]
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
        train_state = self.make_train_state()
        test_best_metric = 1e8 if 'loss' in self.metric else 0
        global_step = 0
        negid = -1
        if 'null' in self.model.tag2id:
            negid = self.model.tag2id['null']
        if negid == -1:
            raise Exception("negative tag not is 'null'")

        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
            train_state['epoch_index'] = epoch
            preds_kvpairs = []
            golds_kvpairs = []
            avg_loss = Mean()
            avg_start_acc = Mean()
            avg_end_acc = Mean()
            prec = Mean()
            rec = Mean()
            for ith, data in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                args = data[3:-2] + data[-1:] # except loss_mask
                seq_out, start_logits, end_logits = self.parallel_model(*args) 
                start_labels = data[0] # (B, S)
                end_labels = data[1] # (B, S)
                attr_labels = data[2]
                loss_mask = data[-2] # (B, S)
                valid_seq_len = loss_mask.sum(dim=-1) # (B)
                inputs_seq, inputs_mask = data[3], data[-1] # (B, S), (B, S)
                inputs_seq_len = inputs_mask.sum(dim=-1) # (B)
                bs = start_labels.size(0)

                # construct span logits and span labels for span loss
                if self.span_bce is not None:
                    span_out = []
                    span_labels = []
                    span_range = []
                    for i in range(bs):
                        span_range.append([])
                        spos, tpos = (inputs_seq_len[i] - valid_seq_len[i] - 1).item(), (inputs_seq_len[i] - 1).item()
                        for j in range(spos, tpos):
                            if start_logits[i][j][1] <= start_logits[i][j][0]:
                                continue
                            for k in range(j, tpos):
                                if end_logits[i][k][1] > end_logits[i][k][0]:
                                    span_out.append(torch.cat([seq_out[i][j], seq_out[i][k]]))
                                    if start_labels[i][j] == 1 and end_labels[i][k] == 1:
                                        span_labels.append(1)
                                    else:
                                        span_labels.append(0)
                                    span_range[-1].append((j - spos, k + 1 - spos))
                    if len(span_out) > 0:
                        span_out = torch.stack(span_out, dim=0)
                        span_logits = self.model.span_fc(span_out).squeeze(dim=-1)
                        span_labels = torch.tensor(span_labels).float().to(span_logits.device)

                # loss and optimize
                start_loss = self.criterion(start_logits.permute(0, 2, 1), start_labels)
                start_loss = (torch.sum(start_loss * loss_mask, dim=-1) / valid_seq_len).mean()
                end_loss = self.criterion(end_logits.permute(0, 2, 1), end_labels)
                end_loss = (torch.sum(end_loss * loss_mask, dim=-1) / valid_seq_len).mean()
                if self.span_bce is not None and len(span_out) > 0:
                    span_loss = self.span_bce(span_logits, span_labels)
                if self.autoweighted_loss is not None:
                    if self.span_bce is not None and len(span_out) > 0:
                        loss = self.autoweighted_loss(start_loss, end_loss, span_loss)
                    else:
                        loss = self.autoweighted_loss(start_loss, end_loss)
                else:
                    if self.span_bce is not None and len(span_out) > 0:
                        loss = (start_loss + end_loss + span_loss) / 3
                    else:
                        loss = (start_loss + end_loss) / 2
                loss.backward()
                if self.adv is not None:
                    loss = adversarial_perturbation_mrc_span_mtl(self.adv, self.parallel_model, self.criterion, self.span_bce, self.autoweighted_loss, 3, 0., start_labels, end_labels, *data[3:])
                # torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.warmup_step > 0:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                # preds
                start_preds = start_logits.argmax(dim=-1)
                end_preds = end_logits.argmax(dim=-1)
                if self.span_bce is not None and len(span_out) > 0:
                    span_preds = (torch.sigmoid(span_logits) >= 0.5).long()

                # get token sequence
                start_preds = start_preds.detach().cpu().numpy()
                end_preds = end_preds.detach().cpu().numpy()
                start_labels = start_labels.detach().cpu().numpy()
                end_labels = end_labels.detach().cpu().numpy()
                attr_labels = attr_labels.detach().cpu().numpy()
                inputs_seq = inputs_seq.detach().cpu().numpy()
                inputs_seq_len = inputs_seq_len.detach().cpu().numpy()
                loss_mask = loss_mask.detach().cpu().numpy()
                valid_seq_len = valid_seq_len.detach().cpu().numpy()
                span_cnt = 0
                attr_labels_name = [self.model.id2tag[tid] for tid in attr_labels]
                for i in range(bs):
                    spos, tpos = inputs_seq_len[i] - valid_seq_len[i] - 1, inputs_seq_len[i] - 1
                    if self.span_bce is None:
                        start_pred_seq = [attr_labels_name[i] if tid else 'null' for tid in start_preds[i][spos:tpos]]
                        end_pred_seq = [attr_labels_name[i] if tid else 'null' for tid in end_preds[i][spos:tpos]]
                    start_gold_seq = [attr_labels_name[i] if tid else 'null' for tid in start_labels[i][spos:tpos]]
                    end_gold_seq = [attr_labels_name[i] if tid else 'null' for tid in end_labels[i][spos:tpos]]
                    char_seq = [self.model.sequence_encoder.tokenizer.convert_ids_to_tokens(int(tid)) for tid in inputs_seq[i][spos:tpos]]
                    # print(char_seq)
                    # print(start_gold_seq)
                    # print(end_gold_seq)
                    assert len(char_seq) == len(start_gold_seq) == len(end_pred_seq)
                    if self.span_bce is not None:
                        if len(span_out) > 0:
                            pred_kvpairs = [(span_range[i][j], attr_labels_name[i], ''.join(char_seq[span_range[i][j][0]:span_range[i][j][1]])) for j in range(len(span_range[i])) if span_preds[span_cnt+j]]
                            span_cnt += len(span_range[i])
                        else:
                            pred_kvpairs = []
                    else:
                        pred_kvpairs = extract_kvpairs_by_start_end(start_pred_seq, end_pred_seq, char_seq, self.model.id2tag[negid])
                    gold_kvpairs = extract_kvpairs_by_start_end(start_gold_seq, end_gold_seq, char_seq, self.model.id2tag[negid])
                    preds_kvpairs.append(pred_kvpairs)
                    golds_kvpairs.append(gold_kvpairs)

                # metrics update
                p_sum = 0
                r_sum = 0
                hits = 0
                for pred, gold in zip(preds_kvpairs[-bs:], golds_kvpairs[-bs:]):
                    p_sum += len(pred)
                    r_sum += len(gold)
                    for label in pred:
                        if label in gold:
                            hits += 1
                start_acc = ((start_labels == start_preds) * (start_labels != 0) * loss_mask).sum()
                end_acc = ((end_labels == end_preds) * (end_labels != 0) * loss_mask).sum()
                avg_loss.update(loss.item() * bs, bs)
                avg_start_acc.update(start_acc, ((start_labels != 0) * loss_mask).sum())
                avg_end_acc.update(end_acc, ((end_labels != 0) * loss_mask).sum())
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)

                # Log
                global_step += 1
                if global_step % 20 == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.logger.info(f'Training...Epoches: {epoch}, steps: {global_step}, loss: {avg_loss.avg:.4f}, start_acc: {avg_start_acc.avg:.4f}, end_acc: {avg_end_acc.avg:.4f}, micro_p: {prec.avg:.4f}, micro_r: {rec.avg:.4f}, micro_f1: {micro_f1:.4f}')

                # tensorboard training writer
                if global_step % 5 == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.writer.add_scalar('train loss', avg_loss.avg, global_step=global_step)
                    self.writer.add_scalar('train start acc', avg_start_acc.avg, global_step=global_step)
                    self.writer.add_scalar('train end acc', avg_end_acc.avg, global_step=global_step)
                    self.writer.add_scalar('train micro precision', prec.avg, global_step=global_step)
                    self.writer.add_scalar('train micro recall', rec.avg, global_step=global_step)
                    self.writer.add_scalar('train micro f1', micro_f1, global_step=global_step)
            micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
            train_state['train_metrics'].append({'loss': avg_loss.avg, 'start_acc': avg_start_acc.avg, 'end_acc': avg_end_acc.avg, 
                                                'micro_p': prec.avg, 'micro_r': rec.avg, 'micro_f1': micro_f1})

            # Val 
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader) 
            self.logger.info(f'Evaluation result: {result}.')
            self.logger.info('Metric {} current / best: {} / {}'.format(self.metric, result[self.metric], train_state['early_stopping_best_val']))
            category_result = result.pop('category-p/r/f1')
            train_state['val_metrics'].append(result)
            result['category-p/r/f1'] = category_result
            self.update_train_state(train_state)
            if self.early_stopping_step > 0:
                if not self.warmup_step > 0:
                    self.scheduler.step(train_state['val_metrics'][-1][self.metric])
                if train_state['stop_early']:
                    break
            
            # tensorboard val writer
            self.writer.add_scalar('val loss', result['loss'], epoch)
            self.writer.add_scalar('val start acc', result['start_acc'], epoch)
            self.writer.add_scalar('val end acc', result['end_acc'], epoch)
            self.writer.add_scalar('val micro precision', result['micro_p'], epoch)
            self.writer.add_scalar('val micro recall', result['micro_r'], epoch)
            self.writer.add_scalar('val micro f1', result['micro_f1'], epoch)
            
            # test
            if hasattr(self, 'test_loader') and 'msra' not in self.ckpt:
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
                    self.save_model(self.ckpt[:-10] + '_test' + self.ckpt[-10:])
                    test_best_metric = result[self.metric]

        self.logger.info("Best %s on val set: %f" % (self.metric, train_state['early_stopping_best_val']))
        if hasattr(self, 'test_loader') and 'msra' not in self.ckpt:
            self.logger.info("Best %s on test set: %f" % (self.metric, test_best_metric))


    def eval_model(self, eval_loader):
        self.eval()
        preds_kvpairs = []
        golds_kvpairs = []
        category_result = defaultdict(lambda: [0, 0, 0]) # gold, pred, correct
        avg_loss = Mean()
        avg_start_acc = Mean()
        avg_end_acc = Mean()
        prec = Mean()
        rec = Mean()
        if 'null' in self.model.tag2id:
            negid = self.model.tag2id['null']
        if negid == -1:
            raise Exception("negative tag not in 'null'")
        with torch.no_grad():
            for ith, data in enumerate(eval_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                args = data[3:-2] + data[-1:]
                seq_out, start_logits, end_logits = self.parallel_model(*args)
                start_labels = data[0] # (B, S)
                end_labels = data[1] # (B, S)
                attr_labels = data[2]
                loss_mask = data[-2] # (B, S)
                valid_seq_len = loss_mask.sum(dim=-1) # (B)
                inputs_seq, inputs_mask = data[3], data[-1] # (B, S), (B, S)
                inputs_seq_len = inputs_mask.sum(dim=-1) # (B)
                bs = start_labels.size(0)

                # construct span logits and span labels for span loss
                if self.span_bce is not None:
                    span_out = []
                    span_range = []
                    for i in range(bs):
                        span_range.append([])
                        spos, tpos = (inputs_seq_len[i] - valid_seq_len[i] - 1).item(), (inputs_seq_len[i] - 1).item()
                        for j in range(spos, tpos):
                            if start_logits[i][j][1] <= start_logits[i][j][0]:
                                continue
                            for k in range(j, tpos):
                                if end_logits[i][k][1] > end_logits[i][k][0]:
                                    span_out.append(torch.cat([seq_out[i][j], seq_out[i][k]]))
                                    span_range[-1].append((j - spos, k + 1 - spos))
                    if len(span_out) > 0:
                        span_out = torch.stack(span_out, dim=0)
                        span_logits = self.model.span_fc(span_out).float().squeeze(dim=-1)

                # loss
                start_loss = self.criterion(start_logits.permute(0, 2, 1), start_labels)
                start_loss = (torch.sum(start_loss * loss_mask, dim=-1) / valid_seq_len).mean()
                end_loss = self.criterion(end_logits.permute(0, 2, 1), end_labels)
                end_loss = (torch.sum(end_loss * loss_mask, dim=-1) / valid_seq_len).mean()
                if self.span_bce is not None and len(span_out) > 0:
                    span_loss = self.span_bce(span_logits, span_labels)
                if self.autoweighted_loss is not None:
                    if self.span_bce is not None and len(span_out) > 0:
                        loss = self.autoweighted_loss(start_loss, end_loss, span_loss)
                    else:
                        loss = self.autoweighted_loss(start_loss, end_loss)
                else:
                    if self.span_bce is not None and len(span_out) > 0:
                        loss = (start_loss + end_loss + span_loss) / 3
                    else:
                        loss = (start_loss + end_loss) / 2

                # preds
                start_preds = start_logits.argmax(dim=-1)
                end_preds = end_logits.argmax(dim=-1)
                if self.span_bce is not None and len(span_out) > 0:
                    span_preds = (torch.sigmoid(span_logits) >= 0.5).long()

                # get token sequence
                start_preds = start_preds.detach().cpu().numpy()
                end_preds = end_preds.detach().cpu().numpy()
                start_labels = start_labels.detach().cpu().numpy()
                end_labels = end_labels.detach().cpu().numpy()
                attr_labels = attr_labels.detach().cpu().numpy()
                inputs_seq = inputs_seq.detach().cpu().numpy()
                inputs_seq_len = inputs_seq_len.detach().cpu().numpy()
                loss_mask = loss_mask.detach().cpu().numpy()
                valid_seq_len = valid_seq_len.detach().cpu().numpy()
                span_cnt = 0
                attr_labels_name = [self.model.id2tag[tid] for tid in attr_labels]
                for i in range(bs):
                    spos, tpos = inputs_seq_len[i] - valid_seq_len[i] - 1, inputs_seq_len[i] - 1
                    if self.span_bce is None:
                        start_pred_seq = [attr_labels_name[i] if tid else 'null' for tid in start_preds[i][spos:tpos]]
                        end_pred_seq = [attr_labels_name[i] if tid else 'null' for tid in end_preds[i][spos:tpos]]
                    start_gold_seq = [attr_labels_name[i] if tid else 'null' for tid in start_labels[i][spos:tpos]]
                    end_gold_seq = [attr_labels_name[i] if tid else 'null' for tid in end_labels[i][spos:tpos]]
                    char_seq = [self.model.sequence_encoder.tokenizer.convert_ids_to_tokens(int(tid)) for tid in inputs_seq[i][spos:tpos]]
                    if self.span_bce is not None:
                        if len(span_out) > 0:
                            pred_kvpairs = [(span_range[i][j], attr_labels_name[i], ''.join(char_seq[span_range[i][j][0]:span_range[i][j][1]])) for j in range(len(span_range[i])) if span_preds[span_cnt+j]]
                            span_cnt += len(span_range[i])
                        else:
                            pred_kvpairs = []
                    else:
                        pred_kvpairs = extract_kvpairs_by_start_end(start_pred_seq, end_pred_seq, char_seq, self.model.id2tag[negid])
                    gold_kvpairs = extract_kvpairs_by_start_end(start_gold_seq, end_gold_seq, char_seq, self.model.id2tag[negid])
                    # hits = 0
                    # for triple in pred_kvpairs:
                    #     if triple in gold_kvpairs:
                    #         hits += 1
                    # if hits != len(pred_kvpairs):
                    #     self.logger.info(f"sentence: {''.join(char_seq)}")
                    #     self.logger.info(f'pred_kvpairs: {pred_kvpairs}')
                    #     # self.logger.info(f'pred_kvpairs: {list(zip(start_pred_seq, end_pred_seq, char_seq))}')
                    #     self.logger.info(f'gold_kvpairs: {gold_kvpairs}')

                    preds_kvpairs.append(pred_kvpairs)
                    golds_kvpairs.append(gold_kvpairs)

                # metrics update
                p_sum = 0
                r_sum = 0
                hits = 0
                for pred, gold in zip(preds_kvpairs[-bs:], golds_kvpairs[-bs:]):
                    for triple in gold:
                        category_result[triple[1]][0] += 1
                    for triple in pred:
                        category_result[triple[1]][1] += 1
                    p_sum += len(pred)
                    r_sum += len(gold)
                    for triple in pred:
                        if triple in gold:
                            hits += 1
                            category_result[triple[1]][2] += 1
                start_acc = ((start_labels == start_preds) * (start_labels != 0) * loss_mask).sum()
                end_acc = ((end_labels == end_preds) * (end_labels != 0) * loss_mask).sum()
                avg_start_acc.update(start_acc, ((start_labels != 0) * loss_mask).sum())
                avg_end_acc.update(end_acc, ((end_labels != 0) * loss_mask).sum())
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)
                avg_loss.update(loss.item() * bs, bs)

                # Log
                if (ith + 1) % 20 == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.logger.info(f'Evaluation...Batches: {ith + 1}, start_acc: {avg_start_acc.avg:.4f}, end_acc: {avg_end_acc.avg:.4f}, micro_p: {prec.avg:.4f}, micro_r: {rec.avg:.4f}, micro_f1: {micro_f1:.4f}')

        for k, v in category_result.items():
            v_golden, v_pred, v_correct = v
            cate_precision = 0 if v_pred == 0 else round(v_correct / v_pred, 4)
            cate_recall = 0 if v_golden == 0 else round(v_correct / v_golden, 4)
            if cate_precision + cate_recall == 0:
                cate_f1 = 0
            else:
                cate_f1 = round(2 * cate_precision * cate_recall / (cate_precision + cate_recall), 4)
            category_result[k] = (cate_precision, cate_recall, cate_f1)
        category_result = {k: v for k, v in sorted(category_result.items(), key=lambda x: x[1][2])}
        p, r, f1 = micro_p_r_f1_score(preds_kvpairs, golds_kvpairs)
        result = {'loss': avg_loss.avg, 'start_acc': avg_start_acc.avg, 'end_acc': avg_end_acc.avg, 'micro_p': p, 'micro_r':r, 'micro_f1':f1, 'category-p/r/f1':category_result}
        return result


    def load_model(self, ckpt):
        state_dict = torch.load(ckpt)
        self.model.load_state_dict(state_dict['model'])
        if self.autoweighted_loss is not None:
            self.autoweighted_loss.load_state_dict(state_dict['autoweighted_loss'])
    

    def save_model(self, ckpt):
        state_dict = {'model': self.model.state_dict()}
        if self.autoweighted_loss is not None:
            state_dict.update({'autoweighted_loss': self.autoweighted_loss.state_dict()})
        torch.save(state_dict, ckpt)
