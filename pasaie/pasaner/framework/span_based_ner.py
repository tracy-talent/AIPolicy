"""
 Author: liujian 
 Date: 2020-11-16 15:21:57 
 Last Modified by: liujian 
 Last Modified time: 2020-11-16 15:21:57 
"""

from ...metrics import Mean, micro_p_r_f1_score
from ...losses import AutomaticWeightedLoss
from ...utils.adversarial import adversarial_perturbation, adversarial_perturbation_span_mtl
from ...utils.entity_extract import extract_kvpairs_by_start_end
from .data_loader import SpanSingleNERDataLoader, SpanMultiNERDataLoader
from .base_framework import BaseFramework

import os
from collections import defaultdict

import torch
from torch import nn


class Span_Single_NER(BaseFramework):
    """span based entity classification, single task"""
    
    def __init__(self, 
                model, 
                train_path, 
                val_path, 
                test_path, 
                ckpt, 
                logger, 
                tb_logdir, 
                max_span=7,
                compress_seq=True,
                tagscheme='bio', 
                batch_size=32, 
                max_epoch=100, 
                lr=1e-3,
                bert_lr=3e-5,
                weight_decay=1e-5,
                early_stopping_step=3,
                warmup_step=300,
                max_grad_norm=5.0,
                metric='micro_f1',
                opt='adam',
                adv='fgm',
                loss='ce',
                dice_alpha=0.6, 
                sampler=None):

        # Load Data
        if train_path != None:
            self.train_loader = SpanSingleNERDataLoader(
                path=train_path,
                tag2id=model.tag2id,
                encoder=model.sequence_encoder,
                batch_size=batch_size,
                shuffle=True,
                compress_seq=compress_seq,
                max_span=max_span,
                sampler=sampler
            )

        if val_path != None:
            self.val_loader = SpanSingleNERDataLoader(
                path=val_path,
                tag2id=model.tag2id,
                encoder=model.sequence_encoder,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq,
                max_span=max_span
            )
        
        if test_path != None:
            self.test_loader = SpanSingleNERDataLoader(
                path=test_path,
                tag2id=model.tag2id,
                encoder=model.sequence_encoder,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq,
                max_span=max_span
            )

        # initialize base class
        super(Span_Single_NER, self).__init__(
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
            adv=adv,
            opt=opt,
            loss=loss,
            loss_weight=self.train_loader.dataset.weight,
            dice_alpha=dice_alpha
        )
        
        self.tagscheme = tagscheme
        

    def train_model(self):
        train_state = self.make_train_state()
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
            avg_loss = Mean()
            avg_acc = Mean()
            prec = Mean()
            rec = Mean()
            for ith, data in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                args = data[1:]
                logits = self.parallel_model(*args)
                labels = data[0]
                span_pos = data[-1]
                bs = labels.size(0)
                preds = logits.argmax(dim=-1)
                
                # Optimize
                if self.adv is None:
                    loss = self.criterion(logits, labels)
                    loss = loss.mean()
                    loss.backward()
                else:
                    loss = adversarial_perturbation(self.adv, self.parallel_model, self.criterion, 3, 0., labels, *args)
                # torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.warmup_step > 0:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                # metrics
                acc = (preds == labels).long().sum().item()
                label_pos = (labels != negid).long().sum().item()
                pred_pos = (preds != negid).long().sum().item()
                true_pos = ((labels == preds).long() * (labels != negid).long()).sum().item()
                avg_loss.update(loss.item(), bs)
                avg_acc.update(acc, bs)
                prec.update(true_pos, pred_pos)
                rec.update(true_pos, label_pos)

                # Log
                global_step += 1
                if global_step % 5 == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.logger.info(f'Training...Epoches: {epoch}, steps: {global_step}, loss: {avg_loss.avg:.4f}, acc: {avg_acc.avg:.4f}, micro_p: {prec.avg:.4f}, micro_r: {rec.avg:.4f}, micro_f1: {micro_f1:.4f}')

                # tensorboard training writer
                if global_step % 5 == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.writer.add_scalar('train loss', avg_loss.avg, global_step=global_step)
                    self.writer.add_scalar('train acc', avg_acc.avg, global_step=global_step)
                    self.writer.add_scalar('train micro precision', prec.avg, global_step=global_step)
                    self.writer.add_scalar('train micro recall', rec.avg, global_step=global_step)
                    self.writer.add_scalar('train micro f1', micro_f1, global_step=global_step)
            micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
            train_state['train_metrics'].append({'loss': avg_loss.avg, 'acc': avg_acc.avg, 'micro_p': prec.avg, 'micro_r': rec.avg, 'micro_f1': micro_f1})

            # refresh train dataset
            if epoch != self.max_epoch - 1:
                self.train_loader.dataset.refresh()

            # Val 
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader) 
            self.logger.info(f'Evaluation result: {result}.')
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
            
        self.logger.info("Best %s on val set: %f" % (self.metric, train_state['early_stopping_best_val']))


    def eval_model(self, eval_loader):
        self.eval()
        category_result = defaultdict(lambda: [0, 0, 0]) # gold, pred, correct
        avg_loss = Mean()
        avg_acc = Mean()
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
                args = data[1:]
                logits = self.parallel_model(*args)
                labels = data[0]
                bs = labels.size(0)
                preds = logits.argmax(dim=-1)

                # loss
                loss = self.criterion(logits, labels).sum().item()
                                
                # metrics
                acc = (preds == labels).long().sum().item()
                label_pos = (labels != negid).long().sum().item()
                pred_pos = (preds != negid).long().sum().item()
                true_pos = ((labels == preds).long() * (labels != negid).long()).sum().item()
                avg_acc.update(acc, bs)
                prec.update(true_pos, pred_pos)
                rec.update(true_pos, label_pos)
                avg_loss.update(loss, bs)

                # Log
                if (ith + 1) % 20 == 0:
                    self.logger.info(f'Evaluation...Batches: {ith + 1} finised')
                
                # category result
                labels = labels.detach().cpu().numpy()
                preds = preds.detach().cpu().numpy()
                for j in range(len(labels)):
                    category_result[self.model.id2tag[labels[j]]][0] += 1
                    category_result[self.model.id2tag[preds[j]]][1] += 1
                    if labels[j] == preds[j]:
                        category_result[self.model.id2tag[labels[j]]][2] += 1
        
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
        micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0.
        result = {'loss':avg_loss.avg, 'acc': avg_acc.avg, 'micro_p': prec.avg, 'micro_r':rec.avg, 'micro_f1':micro_f1, 'category-p/r/f1':category_result}
        return result



class Span_Multi_NER(BaseFramework):
    """train multi task for span_start and span_end"""
    
    def __init__(self, 
                model, 
                train_path, 
                val_path, 
                test_path, 
                ckpt, 
                logger, 
                tb_logdir, 
                max_span=7,
                compress_seq=True,
                tagscheme='bio', 
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
                mtl_autoweighted_loss=True,
                dice_alpha=0.6,
                sampler=None):

        # Load Data
        if train_path != None:
            self.train_loader = SpanMultiNERDataLoader(
                path=train_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=True,
                compress_seq=compress_seq,
                sampler=sampler
            )

        if val_path != None:
            self.val_loader = SpanMultiNERDataLoader(
                path=val_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq
            )
        
        if test_path != None:
            self.test_loader = SpanMultiNERDataLoader(
                path=test_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq
            )

        # initialize base class
        super(Span_Multi_NER, self).__init__(
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
            adv=adv,
            opt=opt,
            loss=loss,
            loss_weight=self.train_loader.dataset.weight,
            dice_alpha=dice_alpha
        )

        self.tagscheme = tagscheme
        # Automatic weighted loss for mtl
        self.autoweighted_loss = None
        if mtl_autoweighted_loss:
            self.autoweighted_loss = AutomaticWeightedLoss(2)


    def train_model(self):
        train_state = self.make_train_state()
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
                args = data[2:]
                start_logits, end_logits = self.parallel_model(data[0], *args)
                start_labels = data[0] # (B, S)
                end_labels = data[1] # (B, S)
                inputs_seq, inputs_mask = data[2], data[-1] # (B, S), (B, S)
                inputs_seq_len = inputs_mask.sum(dim=-1) # (B)
                bs = start_labels.size(0)

                # Loss and Optimize
                if self.adv is None:
                    start_loss = self.criterion(start_logits.permute(0, 2, 1), start_labels)
                    start_loss = torch.sum(start_loss * inputs_mask, dim=-1) / inputs_seq_len
                    end_loss = self.criterion(end_logits.permute(0, 2, 1), end_labels)
                    end_loss = torch.sum(end_loss * inputs_mask, dim=-1) / inputs_seq_len
                    if self.autoweighted_loss is not None:
                        loss = self.autoweighted_loss(start_loss, end_loss)
                    else:
                        loss = (start_loss + end_loss) / 2
                    loss = loss.mean()
                    loss.backward()
                else:
                    loss = adversarial_perturbation_span_mtl(self.adv, self.parallel_model, self.criterion, self.autoweighted_loss, 3, 0., start_labels, end_labels, *args)
                # torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.warmup_step > 0:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                # preds
                start_preds = start_logits.argmax(dim=-1)
                end_preds = end_logits.argmax(dim=-1)

                # get token sequence
                start_preds = start_preds.detach().cpu().numpy()
                end_preds = end_preds.detach().cpu().numpy()
                start_labels = start_labels.detach().cpu().numpy()
                end_labels = end_labels.detach().cpu().numpy()
                inputs_seq = inputs_seq.detach().cpu().numpy()
                inputs_mask = inputs_mask.detach().cpu().numpy()
                inputs_seq_len = inputs_seq_len.detach().cpu().numpy()
                spos, tpos = 1, -1
                for i in range(bs):
                    seqlen = inputs_seq_len[i]
                    if not self.is_bert_encoder:
                        spos, tpos = 0, seqlen
                    start_pred_seq = [self.model.id2tag[tid] for tid in start_preds[i][:seqlen][spos:tpos]]
                    end_pred_seq = [self.model.id2tag[tid] for tid in end_preds[i][:seqlen][spos:tpos]]
                    start_gold_seq = [self.model.id2tag[tid] for tid in start_labels[i][:seqlen][spos:tpos]]
                    end_gold_seq = [self.model.id2tag[tid] for tid in end_labels[i][:seqlen][spos:tpos]]
                    char_seq = [self.model.sequence_encoder.tokenizer.convert_ids_to_tokens(int(tid)) for tid in inputs_seq[i][:seqlen][spos:tpos]]

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
                start_acc = ((start_labels == start_preds) * (start_labels != negid) * inputs_mask).sum()
                end_acc = ((end_labels == end_preds) * (end_labels != negid) * inputs_mask).sum()
                avg_loss.update(loss.item() * bs, bs)
                avg_start_acc.update(start_acc, ((start_labels != negid) * inputs_mask).sum())
                avg_end_acc.update(end_acc, ((end_labels != negid) * inputs_mask).sum())
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)

                # Log
                global_step += 1
                if global_step % 5 == 0:
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
            train_state['train_metrics'].append({'loss': avg_loss.avg, 'micro_p': prec.avg, 'micro_r': rec.avg, 'micro_f1': micro_f1})

            # Val 
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader) 
            self.logger.info(f'Evaluation result: {result}.')
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
            self.writer.add_scalar('val start acc', result['start_acc'], epoch)
            self.writer.add_scalar('val end acc', result['end_acc'], epoch)
            self.writer.add_scalar('val micro precision', result['micro_p'], epoch)
            self.writer.add_scalar('val micro recall', result['micro_r'], epoch)
            self.writer.add_scalar('val micro f1', result['micro_f1'], epoch)
            
        self.logger.info("Best %s on val set: %f" % (self.metric, train_state['early_stopping_best_val']))


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
                args = data[2:]
                start_logits, end_logits = self.parallel_model(None, *args)
                start_labels = data[0] # (B, S)
                end_labels = data[1] # (B, S)
                inputs_seq, inputs_mask = data[2], data[-1] # (B, S), (B, S)
                inputs_seq_len = inputs_mask.sum(dim=-1) # (B)
                bs = start_labels.size(0)

                # loss
                start_loss = self.criterion(start_logits.permute(0, 2, 1), start_labels)
                start_loss = torch.sum(start_loss * inputs_mask, dim=-1) / inputs_seq_len
                end_loss = self.criterion(end_logits.permute(0, 2, 1), end_labels)
                end_loss = torch.sum(end_loss * inputs_mask, dim=-1) / inputs_seq_len
                if self.autoweighted_loss is not None:
                    loss = self.autoweighted_loss(start_loss, end_loss)
                else:
                    loss = (start_loss + end_loss) / 2
                loss = loss.sum().item()

                # preds
                start_preds = start_logits.argmax(dim=-1)
                end_preds = end_logits.argmax(dim=-1)
                ## get token sequence
                start_preds = start_preds.detach().cpu().numpy()
                end_preds = end_preds.detach().cpu().numpy()
                start_labels = start_labels.detach().cpu().numpy()
                end_labels = end_labels.detach().cpu().numpy()
                inputs_seq = inputs_seq.detach().cpu().numpy()
                inputs_mask = inputs_mask.detach().cpu().numpy()
                inputs_seq_len = inputs_seq_len.detach().cpu().numpy()
                spos, tpos = 1, -1
                for i in range(bs):
                    seqlen = inputs_seq_len[i]
                    if not self.is_bert_encoder:
                        spos, tpos = 0, seqlen
                    start_pred_seq = [self.model.id2tag[tid] for tid in start_preds[i][:seqlen][spos:tpos]]
                    end_pred_seq = [self.model.id2tag[tid] for tid in end_preds[i][:seqlen][spos:tpos]]
                    start_gold_seq = [self.model.id2tag[tid] for tid in start_labels[i][:seqlen][spos:tpos]]
                    end_gold_seq = [self.model.id2tag[tid] for tid in end_labels[i][:seqlen][spos:tpos]]
                    char_seq = [self.model.sequence_encoder.tokenizer.convert_ids_to_tokens(int(tid)) for tid in inputs_seq[i][:seqlen][spos:tpos]]

                    pred_kvpairs = extract_kvpairs_by_start_end(start_pred_seq, end_pred_seq, char_seq, self.model.id2tag[negid])
                    gold_kvpairs = extract_kvpairs_by_start_end(start_gold_seq, end_gold_seq, char_seq, self.model.id2tag[negid])

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
                start_acc = ((start_labels == start_preds) * (start_labels != negid) * inputs_mask).sum()
                end_acc = ((end_labels == end_preds) * (end_labels != negid) * inputs_mask).sum()
                avg_start_acc.update(start_acc, ((start_labels != negid) * inputs_mask).sum())
                avg_end_acc.update(end_acc, ((end_labels != negid) * inputs_mask).sum())
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)
                avg_loss.update(loss, bs)

                # Log
                if (ith + 1) % 20 == 0:
                    self.logger.info(f'Evaluation...Batches: {ith + 1} finished')

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
        result = {'loss': avg_loss.avg, 'micro_p': p, 'micro_r':r, 'micro_f1':f1, 'category-p/r/f1':category_result}
        return result


    def load_model(self, ckpt):
        state_dict = torch.load(ckpt)
        self.model.load_state_dict(state_dict['model'])
        if self.autoweighted_loss is not None:
            self.autoweighted_loss.load_state_dict(state_dict['autoweighted_loss'])
    
    
    def save_model(self, ckpt):
        state_dict = {'model': self.model.state_dict()}
        if self.autoweighted_loss is not None:
            state_dict.upadte({'autoweighted_loss': self.autoweighted_loss.state_dict()})
        torch.save(state_dict, ckpt)