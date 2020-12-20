"""
 Author: liujian 
 Date: 2020-10-25 14:31:17 
 Last Modified by: liujian 
 Last Modified time: 2020-10-25 14:31:17 
"""

from ...metrics import Mean, micro_p_r_f1_score
from ...utils import extract_kvpairs_in_bio, extract_kvpairs_in_bmoes, extract_kvpairs_in_bioe
from ...utils.adversarial import adversarial_perturbation
from .data_loader import SingleNERDataLoader
from .base_framework import BaseFramework

import os
from collections import defaultdict

import torch
from torch import nn


class Model_CRF(BaseFramework):
    """model(adaptive) + crf decoder"""
    
    def __init__(self, 
                model, 
                train_path, 
                val_path, 
                test_path, 
                ckpt, 
                logger, 
                tb_logdir, 
                compress_seq=True,
                tagscheme='bmoes', 
                batch_size=32, 
                max_epoch=100, 
                lr=1e-3,
                bert_lr=3e-5,
                weight_decay=1e-5,
                early_stopping_step=3,
                warmup_step=300,
                max_grad_norm=5.0,
                metric='micro_f1',
                adv='fgm',
                opt='adam',
                loss='ce',
                dice_alpha=0.6):

        # Load Data
        if train_path != None:
            self.train_loader = SingleNERDataLoader(
                path=train_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=True,
                compress_seq=compress_seq
            )

        if val_path != None:
            self.val_loader = SingleNERDataLoader(
                path=val_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq
            )
        
        if test_path != None:
            self.test_loader = SingleNERDataLoader(
                path=test_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq
            )

        # initialize base class
        super(Model_CRF, self).__init__(
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
            loss_weight=self.train_loader.dataset.weight if hasattr(self, 'train_loader') else None,
            dice_alpha=dice_alpha
        )

        self.tagscheme = tagscheme


    def train_model(self):
        train_state = self.make_train_state()
        global_step = 0
        negid = -1
        if 'O' in self.model.tag2id:
            negid = self.model.tag2id['O']
        if negid == -1:
            raise Exception("negative tag not is 'O'")

        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
            train_state['epoch_index'] = epoch
            preds_kvpairs = []
            golds_kvpairs = []
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
                outputs_seq = data[0]
                inputs_seq, inputs_mask = data[1], data[-1]
                inputs_seq_len = inputs_mask.sum(dim=-1)
                bs = outputs_seq.size(0)

                # Optimize
                if self.adv is None:
                    if self.model.crf is None:
                        loss = self.criterion(logits.permute(0, 2, 1), outputs_seq) # B * S
                        loss = torch.sum(loss * inputs_mask, dim=-1) / inputs_seq_len # B
                    else:
                        log_likelihood = self.model.crf(logits, outputs_seq, mask=inputs_mask, reduction='none')
                        loss = -log_likelihood / inputs_seq_len
                    loss = loss.mean()
                    loss.backward()
                else:
                    loss = adversarial_perturbation(self.adv, self.parallel_model, self.criterion, 3, 0., outputs_seq, *args)
                # torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.warmup_step > 0:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                # prediction/decode
                if self.model.crf is None:
                    preds_seq = logits.argmax(dim=-1) # B * S
                else:
                    preds_seq = self.model.crf.decode(logits, mask=inputs_mask) # List[List[int]]
                    for pred_seq in preds_seq:
                        pred_seq.extend([negid] * (outputs_seq.size(1) - len(pred_seq)))
                    preds_seq = torch.tensor(preds_seq).to(outputs_seq.device) # B * S
                
                # get token sequence
                preds_seq = preds_seq.detach().cpu().numpy()
                outputs_seq = outputs_seq.detach().cpu().numpy()
                inputs_seq = inputs_seq.detach().cpu().numpy()
                inputs_mask = inputs_mask.detach().cpu().numpy()
                inputs_seq_len = inputs_seq_len.detach().cpu().numpy()
                spos, tpos = 1, -1
                for i in range(bs):
                    seqlen = inputs_seq_len[i]
                    if not self.is_bert_encoder:
                        spos, tpos = 0, seqlen
                    pred_seq_tag = [self.model.id2tag[tid] for tid in preds_seq[i][:seqlen][spos:tpos]]
                    gold_seq_tag = [self.model.id2tag[tid] for tid in outputs_seq[i][:seqlen][spos:tpos]]
                    char_seq = [self.model.sequence_encoder.tokenizer.convert_ids_to_tokens(int(tid)) for tid in inputs_seq[i][:seqlen][spos:tpos]]

                    pred_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}')(pred_seq_tag, char_seq)
                    gold_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}')(gold_seq_tag, char_seq)

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
                acc = ((outputs_seq == preds_seq) * (outputs_seq != negid) * inputs_mask).sum()

                # Log
                avg_loss.update(loss.item() * bs, bs) # must call item to split it from tensor graph, otherwise gpu memory will overflow
                avg_acc.update(acc, ((outputs_seq != negid) * inputs_mask).sum())
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)
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
            self.writer.add_scalar('val acc', result['acc'], epoch)
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
        avg_acc = Mean()
        prec = Mean()
        rec = Mean()
        if 'O' in self.model.tag2id:
            negid = self.model.tag2id['O']
        if negid == -1:
            raise Exception("negative tag not in 'O'")
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
                outputs_seq = data[0]
                inputs_seq, inputs_mask = data[1], data[-1]
                inputs_seq_len = inputs_mask.sum(dim=-1)
                bs = outputs_seq.size(0)

                # loss
                if self.model.crf is None:
                    loss = self.criterion(logits.permute(0, 2, 1), outputs_seq) # B * S
                    loss = torch.sum(loss * inputs_mask, dim=-1) / inputs_seq_len # B
                else:
                    log_likelihood = self.model.crf(logits, outputs_seq, mask=inputs_mask, reduction='none')
                    loss = -log_likelihood / inputs_seq_len
                loss = loss.sum().item()

                # prediction/decode
                if self.model.crf is None:
                    preds_seq = logits.argmax(-1) # B * S
                else:
                    preds_seq = self.model.crf.decode(logits, mask=inputs_mask) # List[List[int]]
                    for pred_seq in preds_seq:
                        pred_seq.extend([negid] * (outputs_seq.size(1) - len(pred_seq)))
                    preds_seq = torch.tensor(preds_seq).to(outputs_seq.device) # B * S

                # get token sequence
                preds_seq = preds_seq.detach().cpu().numpy()
                outputs_seq = outputs_seq.detach().cpu().numpy()
                inputs_seq = inputs_seq.detach().cpu().numpy()
                inputs_mask = inputs_mask.detach().cpu().numpy()
                inputs_seq_len = inputs_seq_len.detach().cpu().numpy()
                spos, tpos = 1, -1
                for i in range(bs):
                    seqlen = inputs_seq_len[i]
                    if not self.is_bert_encoder:
                        spos, tpos = 0, seqlen
                    pred_seq_tag = [self.model.id2tag[tid] for tid in preds_seq[i][:seqlen][spos:tpos]]
                    gold_seq_tag = [self.model.id2tag[tid] for tid in outputs_seq[i][:seqlen][spos:tpos]]
                    char_seq = [self.model.sequence_encoder.tokenizer.convert_ids_to_tokens(int(tid)) for tid in inputs_seq[i][:seqlen][spos:tpos]]
                    
                    pred_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}')(pred_seq_tag, char_seq)
                    gold_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}')(gold_seq_tag, char_seq)

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
                acc = ((outputs_seq == preds_seq) * (outputs_seq != negid) * inputs_mask).sum()
                avg_acc.update(acc, ((outputs_seq != negid) * inputs_mask).sum())
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
        result = {'loss': avg_loss.avg, 'acc': avg_acc.avg, 'micro_p': p, 'micro_r': r, 'micro_f1': f1, 'category-p/r/f1':category_result}
        return result