"""
 Author: liujian
 Date: 2020-10-26 18:01:42
 Last Modified by: liujian
 Last Modified time: 2020-10-26 18:01:42
"""

from ...metrics import Mean, micro_p_r_f1_score
from ...losses import AutomaticWeightedLoss
from ...utils.entity_extract import *
from ...utils.adversarial import adversarial_perturbation_span_attr_mtl
from .data_loader import MultiNERDataLoader
from .base_framework import BaseFramework

import os
from collections import defaultdict

import torch
from torch import nn


class MTL_Span_Attr(BaseFramework):
    """model(adaptive) + multitask learning by entity span and entity attr"""
    
    def __init__(self, 
                model, 
                train_path, 
                val_path, 
                test_path, 
                ckpt, 
                logger, 
                tb_logdir, 
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
                opt='sgd',
                loss='ce',
                mtl_autoweighted_loss=True,
                dice_alpha=0.6):

        # Load Data
        if train_path != None:
            self.train_loader = MultiNERDataLoader(
                path=train_path,
                span2id=model.span2id,
                attr2id=model.attr2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=True,
                compress_seq=compress_seq
            )

        if val_path != None:
            self.val_loader = MultiNERDataLoader(
                path=val_path,
                span2id=model.span2id,
                attr2id=model.attr2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq
            )
        
        if test_path != None:
            self.test_loader = MultiNERDataLoader(
                path=test_path,
                span2id=model.span2id,
                attr2id=model.attr2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq
            )

        # initialize base class
        super(MTL_Span_Attr, self).__init__(
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
            loss_weight=None, # weight_span and weigth_attr are different
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
        span_negid = -1
        if 'O' in self.model.span2id:
            span_negid = self.model.span2id['O']
        if span_negid == -1:
            raise Exception("span negative tag not in 'O'")
        attr_negid = self.model.attr2id['null']
        span_eid = self.model.span2id['E']
        span_sid = self.model.span2id['S']

        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
            train_state['epoch_index'] = epoch
            preds_kvpairs = []
            golds_kvpairs = []
            avg_loss = Mean()
            avg_span_acc = Mean()
            avg_attr_acc = Mean()
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
                logits_span, logits_attr = self.parallel_model(data[0], *args)
                outputs_seq_span = data[0]
                outputs_seq_attr = data[1]
                inputs_seq, inputs_mask = data[2], data[-1]
                inputs_seq_len = inputs_mask.sum(dim=-1)
                bs = outputs_seq_span.size(0)

                # loss and optimizer
                if self.adv is None:
                    if self.model.crf_span is None:
                        loss_span = self.criterion(logits_span.permute(0, 2, 1), outputs_seq_span) # B * S
                        loss_span = torch.sum(loss_span * inputs_mask, dim=-1) / inputs_seq_len # B
                    else:
                        log_likelihood = self.model.crf_span(logits_span, outputs_seq_span, mask=inputs_mask, reduction='none') # B
                        loss_span = -log_likelihood / inputs_seq_len # B
                    if self.model.crf_attr is None:
                        loss_attr = self.criterion(logits_attr.permute(0, 2, 1), outputs_seq_attr) # B * S
                        tag_masks = ((outputs_seq_span == span_eid) | (outputs_seq_span == span_sid)).float()
                        # tag_masks = (outputs_seq_attr != attr_negid).float()
                        loss_attr = torch.sum(loss_attr * tag_masks, dim=-1) / torch.sum(tag_masks, dim=-1) # B
                    else:
                        log_likelihood = self.model.crf_attr(logits_attr, outputs_seq_attr, mask=inputs_mask, reduction='none') # B
                        loss_attr = -log_likelihood / inputs_seq_len # B
                    loss_span, loss_attr = loss_span.mean(), loss_attr.mean()
                    if self.autoweighted_loss is not None:
                        loss = self.autoweighted_loss(loss_span, loss_attr)
                    else:
                        loss = (loss_span + loss_attr) / 2
                    loss.backward()
                else:
                    loss = adversarial_perturbation_span_attr_mtl(adv, self.parallel_model, self.criterion, self.autoweighted_loss, 3, 0., outputs_span_out, outputs_attr_out, *data[2:])
                torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.warmup_step > 0:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                # prediction/decode
                if self.model.crf_span is None:
                    preds_seq_span = logits_span.argmax(dim=-1) # B * S
                else:
                    preds_seq_span = self.model.crf_span.decode(logits_span, mask=inputs_mask) # List[List[int]]
                    for pred_seq_span in preds_seq_span:
                        pred_seq_span.extend([span_negid] * (outputs_seq_span.size(1) - len(pred_seq_span)))
                    preds_seq_span = torch.tensor(preds_seq_span).to(outputs_seq_span.device) # B * S
                if self.model.crf_attr is None:
                    preds_seq_attr = logits_attr.argmax(dim=-1) # B * S
                else:
                    preds_seq_attr = self.model.crf_attr.decode(logits_attr, mask=inputs_mask) # List[List[int]]
                    for pred_seq_attr in preds_seq_attr:
                        pred_seq_attr.extend([attr_negid] * (outputs_seq_attr.size(1) - len(pred_seq_attr)))
                    preds_seq_attr = torch.tensor(preds_seq_attr).to(outputs_seq_attr.device) # B * S
                
                # get token sequence
                preds_seq_span = preds_seq_span.detach().cpu().numpy()
                preds_seq_attr = preds_seq_attr.detach().cpu().numpy()
                outputs_seq_span = outputs_seq_span.detach().cpu().numpy()
                outputs_seq_attr = outputs_seq_attr.detach().cpu().numpy()
                inputs_mask = inputs_mask.detach().cpu().numpy()
                inputs_seq = inputs_seq.detach().cpu().numpy()
                inputs_seq_len = inputs_seq_len.detach().cpu().numpy()
                spos, tpos = 1, -1
                for i in range(bs):
                    seqlen = inputs_seq_len[i]
                    if not self.is_bert_encoder:
                        spos, tpos = 0, seqlen
                    pred_seq_span_tag = [self.model.id2span[sid] for sid in preds_seq_span[i][:seqlen][spos:tpos]]
                    gold_seq_span_tag = [self.model.id2span[sid] for sid in outputs_seq_span[i][:seqlen][spos:tpos]]
                    pred_seq_attr_tag = [self.model.id2attr[aid] for aid in preds_seq_attr[i][:seqlen][spos:tpos]]
                    gold_seq_attr_tag = [self.model.id2attr[aid] for aid in outputs_seq_attr[i][:seqlen][spos:tpos]]
                    char_seq = [self.model.sequence_encoder.tokenizer.convert_ids_to_tokens(int(tid)) for tid in inputs_seq[i][:seqlen][spos:tpos]]

                    pred_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}_by_endtag')(pred_seq_span_tag, char_seq, pred_seq_attr_tag)
                    gold_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}_by_endtag')(gold_seq_span_tag, char_seq, gold_seq_attr_tag)

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
                span_acc = ((outputs_seq_span == preds_seq_span) * (outputs_seq_span != span_negid) * inputs_mask).sum()
                attr_acc = ((outputs_seq_attr == preds_seq_attr) * (outputs_seq_attr != attr_negid) * inputs_mask).sum()

                # Log
                avg_loss.update(loss.item() * bs, bs)
                avg_span_acc.update(span_acc, ((outputs_seq_span != span_negid) * inputs_mask).sum())
                avg_attr_acc.update(attr_acc, ((outputs_seq_attr != attr_negid) * inputs_mask).sum())
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)
                global_step += 1
                if global_step % 5 == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.logger.info(f'Training...Epoches: {epoch}, steps: {global_step}, loss: {avg_loss.avg:.4f}, span_acc: {avg_span_acc.avg:.4f}, attr_acc: {avg_attr_acc.avg:.4f}, micro_p: {prec.avg:.4f}, micro_r: {rec.avg:.4f}, micro_f1: {micro_f1:.4f}')

                # tensorboard training writer
                if global_step % 5 == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.writer.add_scalar('train loss', avg_loss.avg, global_step=global_step)
                    self.writer.add_scalar('train span acc', avg_span_acc.avg, global_step=global_step)
                    self.writer.add_scalar('train attr acc', avg_attr_acc.avg, global_step=global_step)
                    self.writer.add_scalar('train micro precision', prec.avg, global_step=global_step)
                    self.writer.add_scalar('train micro recall', rec.avg, global_step=global_step)
                    self.writer.add_scalar('train micro f1', micro_f1, global_step=global_step)
            micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
            train_state['train_metrics'].append({'loss': avg_loss.avg, 'span_acc': avg_span_acc.avg, 'attr_acc': avg_attr_acc.avg,'micro_p': prec.avg, 'micro_r': rec.avg, 'micro_f1': micro_f1})

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
            self.writer.add_scalar('val span acc', result['span_acc'], epoch)
            self.writer.add_scalar('val attr acc', result['attr_acc'], epoch)
            self.writer.add_scalar('val micro precision', result['micro_p'], epoch)
            self.writer.add_scalar('val micro recall', result['micro_r'], epoch)
            self.writer.add_scalar('val micro f1', result['micro_f1'], epoch)
            
        self.logger.info("Best %s on val set: %f" % (self.metric, train_state['early_stopping_best_val']))


    def eval_model(self, eval_loader):
        self.eval()
        span_negid = -1
        if 'O' in self.model.span2id:
            span_negid = self.model.span2id['O']
        if span_negid == -1:
            raise Exception("span negative tag not in 'O'")
        attr_negid = self.model.attr2id['null']
        
        preds_kvpairs = []
        golds_kvpairs = []
        category_result = defaultdict(lambda: [0, 0, 0]) # gold, pred, correct
        avg_loss = Mean()
        avg_span_acc = Mean()
        avg_attr_acc = Mean()
        prec = Mean()
        rec = Mean()
        with torch.no_grad():
            for ith, data in enumerate(eval_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                args = data[2:]
                logits_span, logits_attr = self.parallel_model(None, *args)
                outputs_seq_span = data[0]
                outputs_seq_attr = data[1]
                inputs_seq, inputs_mask = data[2], data[-1]
                inputs_seq_len = inputs_mask.sum(dim=-1)
                bs = outputs_seq_span.size(0)

                # loss
                if self.model.crf_span is None:
                    loss_span = self.criterion(logits_span.permute(0, 2, 1), outputs_seq_span) # B * S
                    loss_span = torch.sum(loss_span * inputs_mask, dim=-1) / inputs_seq_len # B
                else:
                    log_likelihood = self.model.crf_span(logits_span, outputs_seq_span, mask=inputs_mask, reduction='none') # B
                    loss_span = -log_likelihood / inputs_seq_len # B
                if self.model.crf_attr is None:
                    loss_attr = self.criterion(logits_attr.permute(0, 2, 1), outputs_seq_attr) # B * S
                    tag_masks = ((outputs_seq_span == span_eid) | (outputs_seq_span == span_sid)).float()
                    # tag_masks = (outputs_seq_attr != attr_negid).float()
                    loss_attr = torch.sum(loss_attr * tag_masks, dim=-1) / torch.sum(tag_masks, dim=-1) # B
                else:
                    log_likelihood = self.model.crf_attr(logits_attr, outputs_seq_attr, mask=inputs_mask, reduction='none') # B
                    loss_attr = -log_likelihood / inputs_seq_len # B
                loss_span, loss_attr = loss_span.mean(), loss_attr.mean()
                if self.autoweighted_loss is not None:
                    loss = self.autoweighted_loss(loss_span, loss_attr)
                else:
                    loss = (loss_span + loss_attr) / 2
                loss = loss.item()

                # prediction/decode
                if self.model.crf_span is None:
                    preds_seq_span = logits_span.argmax(dim=-1) # B
                else:
                    preds_seq_span = self.model.crf_span.decode(logits_span, mask=inputs_mask) # List[List[int]]
                    for pred_seq_span in preds_seq_span:
                        pred_seq_span.extend([span_negid] * (outputs_seq_span.size(1) - len(pred_seq_span)))
                    preds_seq_span = torch.tensor(preds_seq_span).to(outputs_seq_span.device) # B * S
                _, logits_attr = self.parallel_model(preds_seq_span, *args)
                if self.model.crf_attr is None:
                    preds_seq_attr = logits_attr.argmax(dim=-1) # B
                else:
                    preds_seq_attr = self.model.crf_attr.decode(logits_attr, mask=inputs_mask)
                    for pred_seq_attr in preds_seq_attr:
                        pred_seq_attr.extend([attr_negid] * (outputs_seq_attr.size(1) - len(pred_seq_attr)))
                    preds_seq_attr = torch.tensor(preds_seq_attr).to(outputs_seq_attr.device) # B * S

                # get token sequence
                preds_seq_span = preds_seq_span.detach().cpu().numpy()
                preds_seq_attr = preds_seq_attr.detach().cpu().numpy()
                outputs_seq_span = outputs_seq_span.detach().cpu().numpy()
                outputs_seq_attr = outputs_seq_attr.detach().cpu().numpy()
                inputs_mask = inputs_mask.detach().cpu().numpy()
                inputs_seq = inputs_seq.detach().cpu().numpy()
                inputs_seq_len = inputs_seq_len.detach().cpu().numpy()
                spos, tpos = 1, -1
                for i in range(bs):
                    seqlen = inputs_seq_len[i]
                    if not self.is_bert_encoder:
                        spos, tpos = 0, seqlen
                    pred_seq_span_tag = [self.model.id2span[sid] for sid in preds_seq_span[i][:seqlen][spos:tpos]]
                    gold_seq_span_tag = [self.model.id2span[sid] for sid in outputs_seq_span[i][:seqlen][spos:tpos]]
                    pred_seq_attr_tag = [self.model.id2attr[aid] for aid in preds_seq_attr[i][:seqlen][spos:tpos]]
                    gold_seq_attr_tag = [self.model.id2attr[aid] for aid in outputs_seq_attr[i][:seqlen][spos:tpos]]
                    char_seq = [self.model.sequence_encoder.tokenizer.convert_ids_to_tokens(int(cid)) for cid in inputs_seq[i][:seqlen][spos:tpos]]

                    pred_seq_tag = [span + '-' + attr if span != 'O' else 'O' for span, attr in zip(pred_seq_span_tag, pred_seq_attr_tag)]
                    gold_seq_tag = [span + '-' + attr if span != 'O' else 'O' for span, attr in zip(gold_seq_span_tag, gold_seq_attr_tag)]
                    pred_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}_by_endtag')(pred_seq_span_tag, char_seq, pred_seq_attr_tag)
                    gold_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}_by_endtag')(gold_seq_span_tag, char_seq, gold_seq_attr_tag)

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
                span_acc = ((outputs_seq_span == preds_seq_span) * (outputs_seq_span != span_negid) * inputs_mask).sum()
                attr_acc = ((outputs_seq_attr == preds_seq_attr) * (outputs_seq_attr != attr_negid) * inputs_mask).sum()
                avg_span_acc.update(span_acc, ((outputs_seq_span != span_negid) * inputs_mask).sum())
                avg_attr_acc.update(attr_acc, ((outputs_seq_attr != attr_negid) * inputs_mask).sum())
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)
                avg_loss.update(loss * bs, bs)

                # log
                if (ith + 1) % 10 == 0:
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
        result = {'loss': avg_loss.avg, 'span_acc': avg_span_acc.avg, 'attr_acc': avg_attr_acc.avg, 'micro_p': p, 'micro_r':r, 'micro_f1':f1, 'category-p/r/f1':category_result}
        return result