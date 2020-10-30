"""
 Author: liujian
 Date: 2020-10-26 18:01:42
 Last Modified by: liujian
 Last Modified time: 2020-10-26 18:01:42
"""

from ...metrics import Mean, micro_p_r_f1_score
from ...utils import *
from .data_loader import MultiNERDataLoader

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import os


class MTL_Span_Attr(nn.Module):
    """model(adaptive) + multitask learning by entity span and entity attr"""
    
    def __init__(self, 
                model, 
                train_path, 
                val_path, 
                test_path, 
                ckpt, 
                logger, 
                tb_logdir, 
                tagscheme='bio',
                batch_size=32, 
                max_epoch=100, 
                lr=1e-3,
                bert_lr=3e-5,
                weight_decay=1e-5,
                warmup_step=300,
                opt='sgd'):

        super(MTL_Span_Attr, self).__init__()
        if 'bert' in model.sequence_encoder.__class__.__name__.lower():
            self.is_bert_encoder = True
        else:
            self.is_bert_encoder = False
        self.max_epoch = max_epoch
        self.tagscheme = tagscheme

        # Load Data
        if train_path != None:
            self.train_loader = MultiNERDataLoader(
                path=train_path,
                span2id=model.span2id,
                attr2id=model.attr2id,
                tokenizer=model.sequence_encoder.tokenize,
                is_bert_encoder=self.is_bert_encoder,
                batch_size=batch_size,
                shuffle=True
            )

        if val_path != None:
            self.val_loader = MultiNERDataLoader(
                path=val_path,
                span2id=model.span2id,
                attr2id=model.attr2id,
                tokenizer=model.sequence_encoder.tokenize,
                is_bert_encoder=self.is_bert_encoder,
                batch_size=batch_size,
                shuffle=False
            )
        
        if test_path != None:
            self.test_loader = MultiNERDataLoader(
                path=test_path,
                span2id=model.span2id,
                attr2id=model.attr2id,
                tokenizer=model.sequence_encoder.tokenize,
                is_bert_encoder=self.is_bert_encoder,
                batch_size=batch_size,
                shuffle=False
            )

        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
        # Params and optimizer
        self.lr = lr
        self.bert_lr = lr
        if self.is_bert_encoder:
            encoder_params =self.parallel_model.module.sequence_encoder.parameters() # self.parallel_model.module equals to self.model
            bert_params_id = list(map(id, encoder_params))
        else:
            bert_params_id = []
            encoder_params = []
        other_params = list(filter(lambda p: id(p) not in bert_params_id, self.parallel_model.parameters()))
        grouped_params = [
            {'params': encoder_params, 'lr':bert_lr},
            {'params': other_params, 'lr':lr}
        ]
        # param_i = 0
        # while param_i < len(grouped_params):
        #     if len(grouped_params[param_i]['params']) > 0:
        #         param_i += 1
        #     else:
        #         grouped_params.pop(param_i)
        if opt == 'sgd':
            self.optimizer = optim.SGD(grouped_params, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(grouped_params) # adam weight_decay is not reasonable
        elif opt == 'adamw': # Optimizer for BERT
            # from transformers import AdamW 
            from torch.optim import AdamW # torch 1.6 begin providing AdamW
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
            # param_i = 0
            # while param_i < len(adamw_grouped_params):
            #     if len(adamw_grouped_params[param_i]['params']) > 0:
            #         param_i += 1
            #     else:
            #         adamw_grouped_params.pop(param_i)
            self.optimizer = AdamW(adamw_grouped_params, correct_bias=True) # original: correct_bias=False
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = len(self.train_loader) // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt
        # logger
        self.logger = logger
        # tensorboard writer
        self.writer = SummaryWriter(tb_logdir)


    def train_model(self, metric='micro_f1'):
        best_metric = 0
        global_step = 0
        neg_spanid = -1
        if 'O' in self.model.span2id:
            neg_spanid = self.model.span2id['O']
        if neg_spanid == -1:
            raise Exception("span negative tag not in 'O'")
        neg_attrid = self.train_loader.dataset.neg_attrid

        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
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
                logits_span, logits_attr = self.parallel_model(*args)
                outputs_seq_span = data[0]
                outputs_seq_attr = data[1]
                inputs_seq, inputs_mask = data[2], data[-1]
                inputs_seq_len = inputs_mask.sum(dim=-1)
                bs = outputs_seq_span.size(0)

                if self.model.crf_span is None:
                    loss_span = self.criterion(logits_span.permute(0, 2, 1), outputs_seq_span) # B * S
                    loss_span = torch.sum(loss * inputs_mask, dim=-1) / inputs_seq_len # B
                    preds_seq_span = logits_span.argmax(dim=-1) # B * S
                else:
                    log_likelihood = self.model.crf_span(logits_span, outputs_seq_span, mask=inputs_mask, reduction='none') # B
                    loss_span = -log_likelihood / inputs_seq_len # B
                    preds_seq_span = self.model.crf_span.decode(logits_span, mask=inputs_mask) # List[List[int]]
                    for pred_seq_span in preds_seq_span:
                        pred_seq_span.extend([neg_spanid] * (outputs_seq_span.size(1) - len(pred_seq_span)))
                    preds_seq_span = torch.tensor(preds_seq_span).to(outputs_seq_span.device) # B * S

                if self.model.crf_attr is None:
                    loss_attr = self.criterion(logits_attr.permute(0, 2, 1), outputs_seq_attr) # B * S
                    loss_attr = torch.sum(loss * inputs_mask, dim=-1) / inputs_seq_len # B
                    preds_seq_attr = logits_attr.argmax(dim=-1) # B * S
                else:
                    log_likelihood = self.model.crf_attr(logits_attr, outputs_seq_attr, mask=inputs_mask, reduction='none') # B
                    loss_attr = -log_likelihood / inputs_seq_len # B
                    preds_seq_attr = self.model.crf_attr.decode(logits_attr, mask=inputs_mask) # List[List[int]]
                    for pred_seq_attr in preds_seq_attr:
                        pred_seq_attr.extend([neg_attrid] * (outputs_seq_attr.size(1) - len(pred_seq_attr)))
                    preds_seq_attr = torch.tensor(preds_seq_attr).to(outputs_seq_attr.device) # B * S
                loss = loss_span + loss_attr # 多任务loss汇总

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

                    pred_seq_tag = [span + '-' + attr if span != 'O' else 'O' for span, attr in zip(pred_seq_span_tag, pred_seq_attr_tag)]
                    gold_seq_tag = [span + '-' + attr if span != 'O' else 'O' for span, attr in zip(gold_seq_span_tag, gold_seq_attr_tag)]
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
                span_acc = ((outputs_seq_span == preds_seq_span) * (outputs_seq_span != neg_spanid) * inputs_mask).sum()
                attr_acc = ((outputs_seq_attr == preds_seq_attr) * (outputs_seq_attr != neg_attrid) * inputs_mask).sum()
                loss = loss.mean()

                # Log
                avg_loss.update(loss, bs)
                avg_span_acc.update(span_acc, ((outputs_seq_span != neg_spanid) * inputs_mask).sum())
                avg_attr_acc.update(attr_acc, ((outputs_seq_attr != neg_attrid) * inputs_mask).sum())
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)
                global_step += 1
                if global_step % 10 == 0:
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

                # Optimize
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
            # Val 
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader) 
            self.logger.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                self.logger.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                os.makedirs(folder_path, exist_ok=True)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]
            
            # tensorboard val writer
            self.writer.add_scalar('val span acc', result['span_acc'], epoch)
            self.writer.add_scalar('val attr acc', result['attr_acc'], epoch)
            self.writer.add_scalar('val micro precision', result['micro_p'], epoch)
            self.writer.add_scalar('val micro recall', result['micro_r'], epoch)
            self.writer.add_scalar('val micro f1', result['micro_f1'], epoch)
            
        self.logger.info("Best %s on val set: %f" % (metric, best_metric))


    def eval_model(self, eval_loader):
        self.eval()
        neg_spanid = -1
        if 'O' in self.model.span2id:
            neg_spanid = self.model.span2id['O']
        if neg_spanid == -1:
            raise Exception("span negative tag not in 'O'")
        neg_attrid = self.train_loader.dataset.neg_attrid
        
        preds_kvpairs = []
        golds_kvpairs = []
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
                logits_span, logits_attr = self.parallel_model(*args)
                outputs_seq_span = data[0]
                outputs_seq_attr = data[1]
                inputs_seq, inputs_mask = data[2], data[-1]
                inputs_seq_len = inputs_mask.sum(dim=-1)
                bs = outputs_seq_span.size(0)

                if self.model.crf_span is None:
                    preds_seq_span = logits_span.argmax(dim=-1) # B
                else:
                    preds_seq_span = self.model.crf_span.decode(logits_span, mask=inputs_mask) # List[List[int]]
                    for pred_seq_span in preds_seq_span:
                        pred_seq_span.extend([neg_spanid] * (outputs_seq_span.size(1) - len(pred_seq_span)))
                    preds_seq_span = torch.tensor(preds_seq_span).to(outputs_seq_span.device) # B * S

                if self.model.crf_attr is None:
                    preds_seq_attr = logits_attr.argmax(dim=-1) # B
                else:
                    preds_seq_attr = self.model.crf_attr.decode(logits_attr, mask=inputs_mask)
                    for pred_seq_attr in preds_seq_attr:
                        pred_seq_attr.extend([neg_attrid] * (outputs_seq_attr.size(1) - len(pred_seq_attr)))
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
                span_acc = ((outputs_seq_span == preds_seq_span) * (outputs_seq_span != neg_spanid) * inputs_mask).sum()
                attr_acc = ((outputs_seq_attr == preds_seq_attr) * (outputs_seq_attr != neg_attrid) * inputs_mask).sum()

                # Log
                avg_span_acc.update(span_acc, ((outputs_seq_span != neg_spanid) * inputs_mask).sum())
                avg_attr_acc.update(attr_acc, ((outputs_seq_attr != neg_attrid) * inputs_mask).sum())
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)
                if (ith + 1) % 10 == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.logger.info(f'Evaluation...Batches: {ith + 1}, span_acc: {avg_span_acc.avg:.4f}, attr_acc: {avg_attr_acc.avg:.4f}, micro_p: {prec.avg:.4f}, micro_r: {rec.avg:.4f}, micro_f1: {micro_f1:.4f}')

        p, r, f1 = micro_p_r_f1_score(preds_kvpairs, golds_kvpairs)
        result = {'span_acc': avg_span_acc.avg, 'attr_acc': avg_attr_acc.avg, 'micro_p': p, 'micro_r':r, 'micro_f1':f1}
        self.logger.info(f'Evaluation result: {result}.')
        return result


    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)