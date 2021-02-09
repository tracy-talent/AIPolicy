"""
 Author: liujian
 Date: 2020-10-26 18:01:42
 Last Modified by: liujian
 Last Modified time: 2020-10-26 18:01:42
"""

from ...losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy, AutomaticWeightedLoss
from ...metrics import Mean, micro_p_r_f1_score, BatchMetric
from ...utils.entity_extract import *
from ...utils.adversarial import FGM, PGD, FreeLB
from ...utils.adversarial import adversarial_perturbation_span_attr_boundary_mtl
from .data_loader import SpanAttrBoundaryNERDataLoader, SpanAttrBoundaryTogetherNERDataLoader

import os
import operator
import datetime
from collections import defaultdict

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


class MTL_Span_Attr_Boundary(nn.Module):
    """model(adaptive) + multitask learning by entity span and entity attr"""
    
    def __init__(self, 
                model, 
                train_path, 
                val_path, 
                test_path, 
                ckpt, 
                logger, 
                tb_logdir,
                word_embedding=None,
                bigram_embedding=None, 
                compress_seq=True,
                tagscheme='bmoes',
                batch_size=32, 
                max_epoch=100,
                crf_lr=1e-3,
                lr=1e-3,
                bert_lr=3e-5,
                weight_decay=1e-2,
                early_stopping_step=3,
                warmup_step=300,
                max_grad_norm=5.0,
                metric='micro_f1',
                opt='sgd',
                loss='ce',
                adv='fgm',
                mtl_autoweighted_loss=True,
                dice_alpha=0.6):

        super(MTL_Span_Attr_Boundary, self).__init__()
        encoder_name = model.sequence_encoder.__class__.__name__.lower()
        if 'bert' in encoder_name:
            self.is_bert_encoder = True
        else:
            self.is_bert_encoder = False
        self.max_epoch = max_epoch
        self.metric = metric
        self.tagscheme = tagscheme
        self.max_grad_norm = max_grad_norm
        self.early_stopping_step = early_stopping_step
        if word_embedding is not None and word_embedding.weight.requires_grad:
            self.word_embedding = nn.Embedding(*word_embedding.weight.size())
            self.word_embedding.weight.data.copy_(word_embedding.weight.data)
            self.word_embedding.weight.requires_grad = word_embedding.weight.requires_grad
            del word_embedding
        else:
            self.word_embedding = word_embedding
        if bigram_embedding is not None and bigram_embedding.weight.requires_grad:
            self.bigram_embedding = nn.Embedding(*bigram_embedding.weight.size())
            self.bigram_embedding.weight.data.copy_(bigram_embedding.weight.data)
            self.bigram_embedding.weight.requires_grad = bigram_embedding.weight.requires_grad
            del bigram_embedding
        else:
            self.bigram_embedding = bigram_embedding
        
        preload = True
        # if 'char' in encoder_name and 'bmes' in encoder_name and ('ontonotes4' in train_path or 'msra' in train_path):
        #     preload = False
        # Load Data
        if train_path != None:
            self.train_loader = SpanAttrBoundaryNERDataLoader(
                path=train_path,
                span2id=model.span2id,
                attr2id=model.attr2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=True,
                compress_seq=compress_seq,
                preload=preload
            )

        if val_path != None:
            self.val_loader = SpanAttrBoundaryNERDataLoader(
                path=val_path,
                span2id=model.span2id,
                attr2id=model.attr2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq,
                preload=preload
            )
        
        if test_path != None:
            self.test_loader = SpanAttrBoundaryNERDataLoader(
                path=test_path,
                span2id=model.span2id,
                attr2id=model.attr2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq,
                preload=preload
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
        # Automatic weighted loss for mtl(submodule must after torch.nn.Module initialize)
        self.autoweighted_loss = None
        if mtl_autoweighted_loss:
            self.autoweighted_loss = AutomaticWeightedLoss(num=3, mode='cls')
        # Params and optimizer
        self.lr = lr
        self.bert_lr = bert_lr
        crf_params = [p for n, p in self.named_parameters() if 'crf' in n]
        crf_params_id = list(map(id, crf_params)) # make crf_params lr=1e-2
        pretrained_params_id = []
        if self.word_embedding is not None and self.word_embedding.weight.requires_grad:
            embedding_params = self.word_embedding.parameters()
            pretrained_params_id.extend(list(map(id, embedding_params)))
        if self.bigram_embedding is not None and self.bigram_embedding.weight.requires_grad:
            embedding_params = self.bigram_embedding.parameters()
            pretrained_params_id.extend(list(map(id, embedding_params)))
        if self.is_bert_encoder:
            encoder_params = self.parallel_model.module.sequence_encoder.parameters()
            pretrained_params_id.extend(list(map(id, encoder_params)))
        pretrained_params = list(filter(lambda p: id(p) in pretrained_params_id, self.parameters()))
        other_params = list(filter(lambda p: id(p) not in pretrained_params_id and id(p) not in crf_params_id, self.parameters()))
        other_params_id = list(map(id, other_params))
        grouped_params = [
            {'params': pretrained_params, 'lr': bert_lr},
            {'params': crf_params, 'lr': crf_lr},
            {'params': other_params, 'lr': lr}
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
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay) and id(p) in pretrained_params_id], 
                    'weight_decay': weight_decay,
                    'lr': bert_lr,
                },
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay) and id(p) in crf_params_id], 
                    'weight_decay': weight_decay,
                    'lr': 1e-3,
                },
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay) and id(p) in other_params_id], 
                    'weight_decay': weight_decay,
                    'lr': lr,
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay) and id(p) in pretrained_params_id], 
                    'weight_decay': 0.0,
                    'lr': bert_lr,
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay) and id(p) in crf_params_id], 
                    'weight_decay': 0.0,
                    'lr': 1e-3,
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay) and id(p) in other_params_id], 
                    'weight_decay': 0.0,
                    'lr': lr,
                }
            ]
            # adamw_grouped_params = [
            #     {
            #         'params': [p for n, p in params if not any(nd in n for nd in no_decay) and id(p) in pretrained_params_id], 
            #         'weight_decay': weight_decay,
            #         'lr': bert_lr,
            #     },
            #     {
            #         'params': [p for n, p in params if not any(nd in n for nd in no_decay) and id(p) not in pretrained_params_id], 
            #         'weight_decay': weight_decay,
            #         'lr': lr,
            #     },
            #     {
            #         'params': [p for n, p in params if any(nd in n for nd in no_decay) and id(p) in pretrained_params_id], 
            #         'weight_decay': 0.0,
            #         'lr': bert_lr,
            #     },
            #     {
            #         'params': [p for n, p in params if any(nd in n for nd in no_decay) and id(p) not in pretrained_params_id], 
            #         'weight_decay': 0.0,
            #         'lr': lr,
            #     }
            # ]
            self.optimizer = AdamW(adamw_grouped_params, correct_bias=True) # original: correct_bias=False
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        self.warmup_step = warmup_step
        if warmup_step > 0:
            training_steps = len(self.train_loader) // batch_size * self.max_epoch
            # self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=training_steps)
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
        word_embedding = self.word_embedding
        bigram_embedding = self.bigram_embedding
        del self.word_embedding # avoid embedding to cuda
        del self.bigram_embedding # avoid embedding to cuda
        if torch.cuda.is_available():
            self.cuda()
        self.word_embedding = word_embedding
        self.bigram_embedding = bigram_embedding
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
        span_negid = -1
        if 'O' in self.model.span2id:
            span_negid = self.model.span2id['O']
        if span_negid == -1:
            raise Exception("span negative tag not in 'O'")
        attr_negid = self.model.attr2id['null']
        span_eid = self.model.span2id['E']
        span_sid = self.model.span2id['S']
        is_loss_nan = False

        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
            train_state['epoch_index'] = epoch
            span_preds_kvpairs = []
            span_golds_kvpairs = []
            preds_kvpairs = []
            golds_kvpairs = []
            avg_loss = Mean()
            avg_span_acc = Mean()
            avg_attr_start_acc = Mean()
            avg_attr_end_acc = Mean()
            span_prec = Mean()
            span_rec = Mean()
            prec = Mean()
            rec = Mean()
            log_steps = 5
            if len(self.train_loader.dataset) > 1000:
                log_steps = 10
            for ith, data in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            if i == 4 and self.word_embedding is not None:
                                data[i] = self.word_embedding(data[i]).cuda()
                            elif i == 7 and self.bigram_embedding is not None:
                                data[i] = self.bigram_embedding(data[i]).cuda()
                            else:
                                data[i] = data[i].cuda()
                        except:
                            pass
                else:
                    if self.word_embedding is not None and len(data) > 4:
                        data[4] = self.word_embedding(data[4])
                    if self.bigram_embedding is not None and len(data) > 7:
                        data[7] = self.bigram_embedding(data[7])
                args = data[3:]
                if 'StartPrior' in self.model.__class__.__name__:
                    logits_span, logits_attr_start, logits_attr_end = self.parallel_model(data[1], *args)
                else:
                    logits_span, logits_attr_start, logits_attr_end = self.parallel_model(*args)
                outputs_seq_span = data[0]
                outputs_seq_attr_start = data[1]
                outputs_seq_attr_end = data[2]
                inputs_seq, inputs_mask = data[3], data[-1]
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
                    loss_attr_start = self.criterion(logits_attr_start.permute(0, 2, 1), outputs_seq_attr_start) # B * S
                    loss_attr_start = torch.sum(loss_attr_start * inputs_mask, dim=-1) / inputs_seq_len # B
                    loss_attr_end = self.criterion(logits_attr_end.permute(0, 2, 1), outputs_seq_attr_end) # B * S
                    loss_attr_end = torch.sum(loss_attr_end * inputs_mask, dim=-1) / inputs_seq_len # B
                    loss_span, loss_attr_start, loss_attr_end = loss_span.mean(), loss_attr_start.mean(), loss_attr_end.mean()
                    if self.autoweighted_loss is not None:
                        loss = self.autoweighted_loss(loss_span, loss_attr_start, loss_attr_end)
                    else:
                        if torch.abs(loss_span) > 10:
                            loss = (loss_attr_start + loss_attr_end) / 2
                        else:
                            loss = (loss_span + loss_attr_start + loss_attr_end) / 3
                    loss.backward()
                else:
                    retain_graph = False
                    if self.word_embedding is not None and self.word_embedding.weight.requires_grad:
                        retain_graph = True
                    loss = adversarial_perturbation_span_attr_boundary_mtl(self.adv, self.parallel_model, self.criterion, self.autoweighted_loss, 3, 0., outputs_seq_span, outputs_seq_attr_start, outputs_seq_attr_end, retain_graph, *args)
                if loss.isnan() or torch.abs(loss) > 10:
                    #continue
                    is_loss_nan = True
                    break
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
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
                preds_seq_attr_start = logits_attr_start.argmax(dim=-1) # B * S
                preds_seq_attr_end = logits_attr_end.argmax(dim=-1) # B * S

                # get token sequence
                preds_seq_span = preds_seq_span.detach().cpu().numpy()
                preds_seq_attr_start = preds_seq_attr_start.detach().cpu().numpy()
                preds_seq_attr_end = preds_seq_attr_end.detach().cpu().numpy()
                outputs_seq_span = outputs_seq_span.detach().cpu().numpy()
                outputs_seq_attr_start = outputs_seq_attr_start.detach().cpu().numpy()
                outputs_seq_attr_end = outputs_seq_attr_end.detach().cpu().numpy()
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
                    pred_seq_attr_tag_start = [self.model.id2attr[aid] for aid in preds_seq_attr_start[i][:seqlen][spos:tpos]]
                    gold_seq_attr_tag_start = [self.model.id2attr[aid] for aid in outputs_seq_attr_start[i][:seqlen][spos:tpos]]
                    pred_seq_attr_tag_end = [self.model.id2attr[aid] for aid in preds_seq_attr_end[i][:seqlen][spos:tpos]]
                    gold_seq_attr_tag_end = [self.model.id2attr[aid] for aid in outputs_seq_attr_end[i][:seqlen][spos:tpos]]
                    char_seq = [self.model.sequence_encoder.tokenizer.convert_ids_to_tokens(int(tid)) for tid in inputs_seq[i][:seqlen][spos:tpos]]

                    # pred_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}_by_endtag')(pred_seq_span_tag, char_seq, pred_seq_attr_tag)
                    # gold_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}_by_endtag')(gold_seq_span_tag, char_seq, gold_seq_attr_tag)
                    pred_kvpairs = extract_kvpairs_by_start_end(pred_seq_attr_tag_start, pred_seq_attr_tag_end, char_seq, self.model.id2attr[attr_negid])
                    gold_kvpairs = extract_kvpairs_by_start_end(gold_seq_attr_tag_start, gold_seq_attr_tag_end, char_seq, self.model.id2attr[attr_negid])
                    preds_kvpairs.append(pred_kvpairs)
                    golds_kvpairs.append(gold_kvpairs)

                    span_pred_kvpairs = extract_kvpairs_in_bmoes_without_attr(pred_seq_span_tag, char_seq)
                    span_gold_kvpairs = extract_kvpairs_in_bmoes_without_attr(gold_seq_span_tag, char_seq)
                    span_preds_kvpairs.append(span_pred_kvpairs)
                    span_golds_kvpairs.append(span_gold_kvpairs)

                # metrics update
                span_p_sum = 0
                span_r_sum = 0
                span_hits = 0
                for span_pred, span_gold in zip(span_preds_kvpairs[-bs:], span_golds_kvpairs[-bs:]):
                    span_p_sum += len(span_pred)
                    span_r_sum += len(span_gold)
                    for label in span_pred:
                        if label in span_gold:
                            span_hits += 1
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
                attr_start_acc = ((outputs_seq_attr_start == preds_seq_attr_start) * (outputs_seq_attr_start != attr_negid) * inputs_mask).sum()
                attr_end_acc = ((outputs_seq_attr_end == preds_seq_attr_end) * (outputs_seq_attr_end != attr_negid) * inputs_mask).sum()

                # Log
                avg_loss.update(loss.item() * bs, bs)
                avg_span_acc.update(span_acc, ((outputs_seq_span != span_negid) * inputs_mask).sum())
                avg_attr_start_acc.update(attr_start_acc, ((outputs_seq_attr_start != attr_negid) * inputs_mask).sum())
                avg_attr_end_acc.update(attr_end_acc, ((outputs_seq_attr_end != attr_negid) * inputs_mask).sum())
                span_prec.update(span_hits, span_p_sum)
                span_rec.update(span_hits, span_r_sum)
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)
                global_step += 1
                if global_step % log_steps == 0:
                    span_micro_f1 = 2 * span_prec.avg * span_rec.avg / (span_prec.avg + span_rec.avg) if (span_prec.avg + span_rec.avg) > 0 else 0
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.logger.info(f'Training...Epoches: {epoch}, steps: {global_step}, loss: {avg_loss.avg:.4f}, span_acc: {avg_span_acc.avg:.4f}, attr_start_acc: {avg_attr_start_acc.avg:.4f}, attr_end_acc: {avg_attr_end_acc.avg:.4f}, span_micro_p: {span_prec.avg:.4f}, span_micro_r: {span_rec.avg:.4f}, span_micro_f1: {span_micro_f1:.4f}, micro_p: {prec.avg:.4f}, micro_r: {rec.avg:.4f}, micro_f1: {micro_f1:.4f}')

                # tensorboard training writer
                if global_step % log_steps == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.writer.add_scalar('train loss', avg_loss.avg, global_step=global_step)
                    self.writer.add_scalar('train span acc', avg_span_acc.avg, global_step=global_step)
                    self.writer.add_scalar('train attr start acc', avg_attr_start_acc.avg, global_step=global_step)
                    self.writer.add_scalar('train attr end acc', avg_attr_end_acc.avg, global_step=global_step)
                    self.writer.add_scalar('train span micro precision', span_prec.avg, global_step=global_step)
                    self.writer.add_scalar('train span micro recall', span_rec.avg, global_step=global_step)
                    self.writer.add_scalar('train span micro f1', span_micro_f1, global_step=global_step)
                    self.writer.add_scalar('train micro precision', prec.avg, global_step=global_step)
                    self.writer.add_scalar('train micro recall', rec.avg, global_step=global_step)
                    self.writer.add_scalar('train micro f1', micro_f1, global_step=global_step)
            if is_loss_nan:
                self.logger.info(f'loss has nan or loss > 10: {loss}')
                break
            micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
            train_state['train_metrics'].append({'loss': avg_loss.avg, 'span_acc': avg_span_acc.avg, 'attr_start_acc': avg_attr_start_acc.avg, 'attr_end_acc': avg_attr_end_acc.avg, 'span_micro_p': span_prec.avg, 'span_micro_r': span_rec.avg, 'span_micro_f1': span_micro_f1, 'micro_p': prec.avg, 'micro_r': rec.avg, 'micro_f1': micro_f1})

            # Val 
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            acc = (str(round(result['span_acc'], 4) * 100), str(round(result['attr_start_acc'], 4) * 100), str(round(result['attr_end_acc'], 4) * 100))
            p = (str(round(result['span_micro_p'], 4) * 100), str(round(result['micro_p'], 4) * 100))
            r = (str(round(result['span_micro_r'], 4) * 100), str(round(result['micro_r'], 4) * 100))
            f1 = (str(round(result['span_micro_f1'], 4) * 100), str(round(result['micro_f1'], 4) * 100))
            self.logger.info(f"acc: ({' / '.join(acc)}), p: ({' / '.join(p)}), r: ({' / '.join(r)}), f1: ({' / '.join(f1)})")
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
            self.writer.add_scalar('val span acc', result['span_acc'], epoch)
            self.writer.add_scalar('val attr start acc', result['attr_start_acc'], epoch)
            self.writer.add_scalar('val attr end acc', result['attr_end_acc'], epoch)
            self.writer.add_scalar('val span micro precision', result['span_micro_p'], epoch)
            self.writer.add_scalar('val span micro recall', result['span_micro_r'], epoch)
            self.writer.add_scalar('val span micro f1', result['span_micro_f1'], epoch)
            self.writer.add_scalar('val micro precision', result['micro_p'], epoch)
            self.writer.add_scalar('val micro recall', result['micro_r'], epoch)
            self.writer.add_scalar('val micro f1', result['micro_f1'], epoch)

            # test
            if hasattr(self, 'test_loader') and 'msra' not in self.ckpt and 'policy' not in self.ckpt:
                self.logger.info("=== Epoch %d test ===" % epoch)
                result = self.eval_model(self.test_loader)
                acc = (str(round(result['span_acc'], 4) * 100), str(round(result['attr_start_acc'], 4) * 100), str(round(result['attr_end_acc'], 4) * 100))
                p = (str(round(result['span_micro_p'], 4) * 100), str(round(result['micro_p'], 4) * 100))
                r = (str(round(result['span_micro_r'], 4) * 100), str(round(result['micro_r'], 4) * 100))
                f1 = (str(round(result['span_micro_f1'], 4) * 100), str(round(result['micro_f1'], 4) * 100))
                self.logger.info(f"acc: ({' / '.join(acc)}), p: ({' / '.join(p)}), r: ({' / '.join(r)}), f1: ({' / '.join(f1)})")
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
        if hasattr(self, 'test_loader') and 'msra' not in self.ckpt and 'policy' not in self.ckpt:
            self.logger.info("Best %s on test set: %f" % (self.metric, test_best_metric))


    def eval_model(self, eval_loader, result_file=None):
        self.eval()
        span_negid = -1
        if 'O' in self.model.span2id:
            span_negid = self.model.span2id['O']
        if span_negid == -1:
            raise Exception("span negative tag not in 'O'")
        attr_negid = self.model.attr2id['null']
        span_eid = self.model.span2id['E']
        span_sid = self.model.span2id['S']
        
        span_preds_kvpairs = []
        span_golds_kvpairs = []
        preds_kvpairs = []
        golds_kvpairs = []
        sentences = []
        category_result = defaultdict(lambda: [0, 0, 0]) # gold, pred, correct
        avg_loss = Mean()
        avg_span_acc = Mean()
        avg_attr_start_acc = Mean()
        avg_attr_end_acc = Mean()
        span_prec = Mean()
        span_rec = Mean()
        prec = Mean()
        rec = Mean()
        with torch.no_grad():
            for ith, data in enumerate(eval_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            if i == 4 and self.word_embedding is not None:
                                data[i] = self.word_embedding(data[i]).cuda()
                            elif i == 7 and self.bigram_embedding is not None:
                                data[i] = self.bigram_embedding(data[i]).cuda()
                            else:
                                data[i] = data[i].cuda()
                        except:
                            pass
                else:
                    if self.word_embedding is not None and len(data) > 4:
                        data[4] = self.word_embedding(data[4])
                    if self.bigram_embedding is not None and len(data) > 7:
                        data[7] = self.bigram_embedding(data[7])
                args = data[3:]
                if 'StartPrior' in self.model.__class__.__name__:
                    logits_span, logits_attr_start, logits_attr_end = self.parallel_model(None, *args)
                else:
                    logits_span, logits_attr_start, logits_attr_end = self.parallel_model(*args)
                outputs_seq_span = data[0]
                outputs_seq_attr_start = data[1]
                outputs_seq_attr_end = data[2]
                inputs_seq, inputs_mask = data[3], data[-1]
                inputs_seq_len = inputs_mask.sum(dim=-1)
                bs = outputs_seq_span.size(0)

                # loss
                if self.model.crf_span is None:
                    loss_span = self.criterion(logits_span.permute(0, 2, 1), outputs_seq_span) # B * S
                    loss_span = torch.sum(loss_span * inputs_mask, dim=-1) / inputs_seq_len # B
                else:
                    log_likelihood = self.model.crf_span(logits_span, outputs_seq_span, mask=inputs_mask, reduction='none') # B
                    loss_span = -log_likelihood / inputs_seq_len # B
                loss_attr_start = self.criterion(logits_attr_start.permute(0, 2, 1), outputs_seq_attr_start) # B * S
                loss_attr_start = torch.sum(loss_attr_start * inputs_mask, dim=-1) / inputs_seq_len # B
                loss_attr_end = self.criterion(logits_attr_end.permute(0, 2, 1), outputs_seq_attr_end) # B * S
                loss_attr_end = torch.sum(loss_attr_end * inputs_mask, dim=-1) / inputs_seq_len # B
                loss_span, loss_attr_start, loss_attr_end = loss_span.mean(), loss_attr_start.mean(), loss_attr_end.mean()
                if self.autoweighted_loss is not None:
                    loss = self.autoweighted_loss(loss_span, loss_attr_start, loss_attr_end)
                else:
                    loss = (loss_span + loss_attr_start + loss_attr_end) / 3
                loss = loss.item()

                # prediction/decode
                if self.model.crf_span is None:
                    preds_seq_span = logits_span.argmax(dim=-1) # B
                else:
                    preds_seq_span = self.model.crf_span.decode(logits_span, mask=inputs_mask) # List[List[int]]
                    for pred_seq_span in preds_seq_span:
                        pred_seq_span.extend([span_negid] * (outputs_seq_span.size(1) - len(pred_seq_span)))
                    preds_seq_span = torch.tensor(preds_seq_span).to(outputs_seq_span.device) # B * S
                preds_seq_attr_start = logits_attr_start.argmax(dim=-1) # B
                if 'StartPrior' in self.model.__class__.__name__:
                    _, _, logits_attr_end  = self.parallel_model(preds_seq_attr_start, *args)
                preds_seq_attr_end = logits_attr_end.argmax(dim=-1) # B

                # get token sequence
                preds_seq_span = preds_seq_span.detach().cpu().numpy()
                preds_seq_attr_start = preds_seq_attr_start.detach().cpu().numpy()
                preds_seq_attr_end = preds_seq_attr_end.detach().cpu().numpy()
                outputs_seq_span = outputs_seq_span.detach().cpu().numpy()
                outputs_seq_attr_start = outputs_seq_attr_start.detach().cpu().numpy()
                outputs_seq_attr_end = outputs_seq_attr_end.detach().cpu().numpy()
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
                    pred_seq_attr_tag_start = [self.model.id2attr[aid] for aid in preds_seq_attr_start[i][:seqlen][spos:tpos]]
                    gold_seq_attr_tag_start = [self.model.id2attr[aid] for aid in outputs_seq_attr_start[i][:seqlen][spos:tpos]]
                    pred_seq_attr_tag_end = [self.model.id2attr[aid] for aid in preds_seq_attr_end[i][:seqlen][spos:tpos]]
                    gold_seq_attr_tag_end = [self.model.id2attr[aid] for aid in outputs_seq_attr_end[i][:seqlen][spos:tpos]]
                    char_seq = [self.model.sequence_encoder.tokenizer.convert_ids_to_tokens(int(tid)) for tid in inputs_seq[i][:seqlen][spos:tpos]]
                    sentences.append(''.join(char_seq))

                    # pred_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}_by_endtag')(pred_seq_span_tag, char_seq, pred_seq_attr_tag)
                    # gold_kvpairs = eval(f'extract_kvpairs_in_{self.tagscheme}_by_endtag')(gold_seq_span_tag, char_seq, gold_seq_attr_tag)
                    pred_kvpairs = extract_kvpairs_by_start_end(pred_seq_attr_tag_start, pred_seq_attr_tag_end, char_seq, self.model.id2attr[attr_negid])
                    gold_kvpairs = extract_kvpairs_by_start_end(gold_seq_attr_tag_start, gold_seq_attr_tag_end, char_seq, self.model.id2attr[attr_negid])
                    preds_kvpairs.append(pred_kvpairs)
                    golds_kvpairs.append(gold_kvpairs)

                    span_pred_kvpairs = extract_kvpairs_in_bmoes_without_attr(pred_seq_span_tag, char_seq)
                    span_gold_kvpairs = extract_kvpairs_in_bmoes_without_attr(gold_seq_span_tag, char_seq)
                    span_preds_kvpairs.append(span_pred_kvpairs)
                    span_golds_kvpairs.append(span_gold_kvpairs)

                # metrics update
                span_p_sum = 0
                span_r_sum = 0
                span_hits = 0
                for span_pred, span_gold in zip(span_preds_kvpairs[-bs:], span_golds_kvpairs[-bs:]):
                    span_p_sum += len(span_pred)
                    span_r_sum += len(span_gold)
                    for label in span_pred:
                        if label in span_gold:
                            span_hits += 1
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
                attr_start_acc = ((outputs_seq_attr_start == preds_seq_attr_start) * (outputs_seq_attr_start != attr_negid) * inputs_mask).sum()
                attr_end_acc = ((outputs_seq_attr_end == preds_seq_attr_end) * (outputs_seq_attr_end != attr_negid) * inputs_mask).sum()
                avg_span_acc.update(span_acc, ((outputs_seq_span != span_negid) * inputs_mask).sum())
                avg_attr_start_acc.update(attr_start_acc, ((outputs_seq_attr_start != attr_negid) * inputs_mask).sum())
                avg_attr_end_acc.update(attr_end_acc, ((outputs_seq_attr_end != attr_negid) * inputs_mask).sum())
                span_prec.update(span_hits, span_p_sum)
                span_rec.update(span_hits, span_r_sum)
                prec.update(hits, p_sum)
                rec.update(hits, r_sum)
                avg_loss.update(loss * bs, bs)

                # log
                if (ith + 1) % 10 == 0:
                    self.logger.info(f'Evaluation...Batches: {ith + 1} finished')

        # write result file
        if result_file is not None:
            with open(result_file, 'w', encoding='utf-8') as resf:
                for sent, pred, gold in zip(sentences, preds_kvpairs, golds_kvpairs):
                    # words = self.model.sequence_encoder.tokenizer.tokenize(sent)
                    resf.write(f'{sent}\n{pred}\n{gold}\n\n')

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
        span_p, span_r, span_f1 = micro_p_r_f1_score(span_preds_kvpairs, span_golds_kvpairs)
        result = {'loss': avg_loss.avg, 'span_acc': avg_span_acc.avg, 'attr_start_acc': avg_attr_start_acc.avg, 'attr_end_acc': avg_attr_end_acc.avg, 'span_micro_p': span_p, 'span_micro_r': span_r, 'span_micro_f1': span_f1, 'micro_p': p, 'micro_r':r, 'micro_f1':f1, 'category-p/r/f1':category_result}
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



class English_MTL_Span_Attr_Boundary(MTL_Span_Attr_Boundary):
    def train_model(self):
        train_state = self.make_train_state()
        test_best_metric = 1e8 if 'loss' in self.metric else 0
        global_step = 0
        span_negid = -1
        if 'O' in self.model.span2id:
            span_negid = self.model.span2id['O']
        if span_negid == -1:
            raise Exception("span negative tag not in 'O'")
        attr_negid = self.model.attr2id['null']
        span_eid = self.model.span2id['E']
        span_sid = self.model.span2id['S']
        is_loss_nan = False

        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
            train_state['epoch_index'] = epoch
            avg_loss = Mean()
            span_batch_metric = BatchMetric(num_classes=max(len(self.model.span2id), 2), ignore_classes=[span_negid])
            batch_metric = BatchMetric(num_classes=max(len(self.model.attr2id), 2), ignore_classes=[attr_negid])
            log_steps = 5
            if len(self.train_loader.dataset) > 1000:
                log_steps = 10
            for ith, data in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            if i == 4 and self.word_embedding is not None:
                                data[i] = self.word_embedding(data[i]).cuda()
                            else:
                                data[i] = data[i].cuda()
                        except:
                            pass
                args = data[3:]
                if 'StartPrior' in self.model.__class__.__name__:
                    logits_span, logits_attr_start, logits_attr_end = self.parallel_model(data[1], *args)
                else:
                    logits_span, logits_attr_start, logits_attr_end = self.parallel_model(*args)
                outputs_seq_span = data[0]
                outputs_seq_attr_start = data[1]
                outputs_seq_attr_end = data[2]
                inputs_seq, inputs_mask = data[3], data[-1]
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
                    loss_attr_start = self.criterion(logits_attr_start.permute(0, 2, 1), outputs_seq_attr_start) # B * S
                    loss_attr_start = torch.sum(loss_attr_start * inputs_mask, dim=-1) / inputs_seq_len # B
                    loss_attr_end = self.criterion(logits_attr_end.permute(0, 2, 1), outputs_seq_attr_end) # B * S
                    loss_attr_end = torch.sum(loss_attr_end * inputs_mask, dim=-1) / inputs_seq_len # B
                
                    loss_span, loss_attr_start, loss_attr_end = loss_span.mean(), loss_attr_start.mean(), loss_attr_end.mean()
                    if self.autoweighted_loss is not None:
                        loss = self.autoweighted_loss(loss_span, loss_attr_start, loss_attr_end)
                    else:
                        loss = (loss_span + loss_attr_start + loss_attr_end) / 3
                    loss.backward()
                else:
                    loss = adversarial_perturbation_span_attr_boundary_mtl(self.adv, self.parallel_model, self.criterion, self.autoweighted_loss, 3, 0., outputs_seq_span, outputs_seq_attr_start, outputs_seq_attr_end, *data[3:])
                if loss.isnan():
                    is_loss_nan = True
                    break
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
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
                preds_seq_attr_start = logits_attr_start.argmax(dim=-1) # B * S
                preds_seq_attr_end = logits_attr_end.argmax(dim=-1) # B * S

                # get token sequence
                preds_seq_span = preds_seq_span.detach().cpu().numpy()
                preds_seq_attr_start = preds_seq_attr_start.detach().cpu().numpy()
                preds_seq_attr_end = preds_seq_attr_end.detach().cpu().numpy()
                outputs_seq_span = outputs_seq_span.detach().cpu().numpy()
                outputs_seq_attr_start = outputs_seq_attr_start.detach().cpu().numpy()
                outputs_seq_attr_end = outputs_seq_attr_end.detach().cpu().numpy()
                inputs_mask = inputs_mask.detach().cpu().numpy()
                inputs_seq = inputs_seq.detach().cpu().numpy()
                inputs_seq_len = inputs_seq_len.detach().cpu().numpy()
                spos, tpos = 1, -1
                for i in range(bs):
                    seqlen = inputs_seq_len[i]
                    if not self.is_bert_encoder:
                        spos, tpos = 0, seqlen
                    # span metrics update
                    span_batch_metric.update(preds_seq_span[i][:seqlen][spos:tpos], outputs_seq_span[i][:seqlen][spos:tpos])

                    # attr metrics update
                    ## get prediction of sequence
                    preds_seq_attr_id = np.zeros_like(preds_seq_attr_start[i][:seqlen][spos:tpos])
                    preds_seq_attr_id = attr_negid
                    preds_seq_attr_sid = preds_seq_attr_start[i][:seqlen][spos:tpos]
                    preds_seq_attr_eid = preds_seq_attr_end[i][:seqlen][spos:tpos]
                    for i, attr_sid in enumerate(preds_seq_attr_sid):
                        if attr_sid == attr_negid:
                            continue
                        for j, attr_eid in enumerate(preds_seq_attr_eid):
                            if j > 0 and preds_seq_attr_sid[i + j] != attr_negid or j + 1 > 10:
                                break
                            if attr_sid == attr_eid:
                                for k in range(i, i + j + 1):
                                    preds_seq_attr_id[k] = attr_sid
                                break
                    ## get goldlabel of sequence
                    gold_seq_attr_id = np.zeros_like(outputs_seq_attr_start[i][:seqlen][spos:tpos])
                    gold_seq_attr_id = attr_negid
                    gold_seq_attr_sid = outputs_seq_attr_start[i][:seqlen][spos:tpos]
                    gold_seq_attr_eid = outputs_seq_attr_end[i][:seqlen][spos:tpos]
                    for i, attr_sid in enumerate(gold_seq_attr_sid):
                        if attr_sid == attr_negid:
                            continue
                        for j, attr_eid in enumerate(gold_seq_attr_eid):
                            if j > 0 and gold_seq_attr_sid[i + j] != attr_negid or j + 1 > 10:
                                break
                            if attr_sid == attr_eid:
                                for k in range(i, i + j + 1):
                                    gold_seq_attr_id[k] = attr_sid
                                break
                    batch_metric.update(preds_seq_attr_id, gold_seq_attr_id)
                    

                # get metrics 
                avg_loss.update(loss.item() * bs, bs)
                cur_loss = avg_loss.avg
                cur_span_acc = span_batch_metric.accuracy()
                cur_span_prec = span_batch_metric.precision(reduction='micro')
                cur_span_rec = span_batch_metric.recall(reduction='micro')
                cur_span_f1 = span_batch_metric.f1_score(reduction='micro')
                cur_acc = batch_metric.accuracy()
                cur_prec = batch_metric.precision(reduction='micro')
                cur_rec = batch_metric.recall(reduction='micro')
                cur_f1 = batch_metric.f1_score(reduction='micro')

                # Log
                global_step += 1
                if global_step % log_steps == 0:
                    self.logger.info(f'Training...Epoches: {epoch}, steps: {global_step}, loss: {cur_loss:.4f}, span_acc: {cur_span_acc:.4f}, span_micro_p: {cur_span_prec:.4f}, span_micro_r: {cur_span_rec:.4f}, span_micro_f1: {cur_span_f1:.4f}, micro_p: {cur_prec:.4f}, micro_r: {cur_rec:.4f}, micro_f1: {cur_f1:.4f}')

                # tensorboard training writer
                if global_step % log_steps == 0:
                    self.writer.add_scalar('train loss', cur_loss, global_step=global_step)
                    self.writer.add_scalar('train span acc', cur_span_acc, global_step=global_step)
                    self.writer.add_scalar('train span micro precision', cur_span_prec, global_step=global_step)
                    self.writer.add_scalar('train span micro recall', cur_span_rec, global_step=global_step)
                    self.writer.add_scalar('train span micro f1', cur_span_f1, global_step=global_step)
                    self.writer.add_scalar('train micro precision', cur_prec, global_step=global_step)
                    self.writer.add_scalar('train micro recall', cur_rec, global_step=global_step)
                    self.writer.add_scalar('train micro f1', cur_f1, global_step=global_step)
            if is_loss_nan:
                self.logger.info(f'loss has nan: {loss}')
                break
            train_state['train_metrics'].append({'loss': cur_loss, 'span_acc': cur_span_acc, 'acc': cur_acc, 'span_micro_p': cur_span_prec, 'span_micro_r': cur_span_rec, 'span_micro_f1': cur_span_f1, 'micro_p': cur_prec, 'micro_r': cur_rec, 'micro_f1': cur_f1})

            # Val 
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            acc = (str(round(result['span_acc'], 4) * 100), str(round(result['acc'], 4) * 100))
            p = (str(round(result['span_micro_p'], 4) * 100), str(round(result['micro_p'], 4) * 100))
            r = (str(round(result['span_micro_r'], 4) * 100), str(round(result['micro_r'], 4) * 100))
            f1 = (str(round(result['span_micro_f1'], 4) * 100), str(round(result['micro_f1'], 4) * 100))
            self.logger.info(f"acc: ({' / '.join(acc)}), p: ({' / '.join(p)}), r: ({' / '.join(r)}), f1: ({' / '.join(f1)})")
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
            self.writer.add_scalar('val span acc', result['span_acc'], epoch)
            self.writer.add_scalar('val span micro precision', result['span_micro_p'], epoch)
            self.writer.add_scalar('val span micro recall', result['span_micro_r'], epoch)
            self.writer.add_scalar('val span micro f1', result['span_micro_f1'], epoch)
            self.writer.add_scalar('val micro precision', result['micro_p'], epoch)
            self.writer.add_scalar('val micro recall', result['micro_r'], epoch)
            self.writer.add_scalar('val micro f1', result['micro_f1'], epoch)

            # test
            if hasattr(self, 'test_loader') and 'msra' not in self.ckpt and 'policy' not in self.ckpt:
                self.logger.info("=== Epoch %d test ===" % epoch)
                result = self.eval_model(self.test_loader)
                acc = (str(round(result['span_acc'], 4) * 100), str(round(result['acc'], 4) * 100))
                p = (str(round(result['span_micro_p'], 4) * 100), str(round(result['micro_p'], 4) * 100))
                r = (str(round(result['span_micro_r'], 4) * 100), str(round(result['micro_r'], 4) * 100))
                f1 = (str(round(result['span_micro_f1'], 4) * 100), str(round(result['micro_f1'], 4) * 100))
                self.logger.info(f"acc: ({' / '.join(acc)}), p: ({' / '.join(p)}), r: ({' / '.join(r)}), f1: ({' / '.join(f1)})")
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
        if hasattr(self, 'test_loader') and 'msra' not in self.ckpt and 'policy' not in self.ckpt:
            self.logger.info("Best %s on test set: %f" % (self.metric, test_best_metric))


    def eval_model(self, eval_loader):
        self.eval()
        span_negid = -1
        if 'O' in self.model.span2id:
            span_negid = self.model.span2id['O']
        if span_negid == -1:
            raise Exception("span negative tag not in 'O'")
        attr_negid = self.model.attr2id['null']
        span_eid = self.model.span2id['E']
        span_sid = self.model.span2id['S']
        avg_loss = Mean()
        span_batch_metric = BatchMetric(num_classes=max(len(self.model.span2id), 2), ignore_classes=[span_negid])
        batch_metric = BatchMetric(num_classes=max(len(self.model.attr2id), 2), ignore_classes=[attr_negid])
        
        with torch.no_grad():
            for ith, data in enumerate(eval_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            if i == 4 and self.word_embedding is not None:
                                data[i] = self.word_embedding(data[i]).cuda()
                            else:
                                data[i] = data[i].cuda()
                        except:
                            pass
                args = data[3:]
                if 'StartPrior' in self.model.__class__.__name__:
                    logits_span, logits_attr_start, logits_attr_end = self.parallel_model(None, *args)
                else:
                    logits_span, logits_attr_start, logits_attr_end = self.parallel_model(*args)
                outputs_seq_span = data[0]
                outputs_seq_attr_start = data[1]
                outputs_seq_attr_end = data[2]
                inputs_seq, inputs_mask = data[3], data[-1]
                inputs_seq_len = inputs_mask.sum(dim=-1)
                bs = outputs_seq_span.size(0)

                # loss
                if self.model.crf_span is None:
                    loss_span = self.criterion(logits_span.permute(0, 2, 1), outputs_seq_span) # B * S
                    loss_span = torch.sum(loss_span * inputs_mask, dim=-1) / inputs_seq_len # B
                else:
                    log_likelihood = self.model.crf_span(logits_span, outputs_seq_span, mask=inputs_mask, reduction='none') # B
                    loss_span = -log_likelihood / inputs_seq_len # B
                loss_attr_start = self.criterion(logits_attr_start.permute(0, 2, 1), outputs_seq_attr_start) # B * S
                loss_attr_start = torch.sum(loss_attr_start * inputs_mask, dim=-1) / inputs_seq_len # B
                loss_attr_end = self.criterion(logits_attr_end.permute(0, 2, 1), outputs_seq_attr_end) # B * S
                loss_attr_end = torch.sum(loss_attr_end * inputs_mask, dim=-1) / inputs_seq_len # B
                loss_span, loss_attr_start, loss_attr_end = loss_span.mean(), loss_attr_start.mean(), loss_attr_end.mean()
                if self.autoweighted_loss is not None:
                    loss = self.autoweighted_loss(loss_span, loss_attr_start, loss_attr_end)
                else:
                    loss = (loss_span + loss_attr_start + loss_attr_end) / 3
                loss = loss.item()

                # prediction/decode
                if self.model.crf_span is None:
                    preds_seq_span = logits_span.argmax(dim=-1) # B
                else:
                    preds_seq_span = self.model.crf_span.decode(logits_span, mask=inputs_mask) # List[List[int]]
                    for pred_seq_span in preds_seq_span:
                        pred_seq_span.extend([span_negid] * (outputs_seq_span.size(1) - len(pred_seq_span)))
                    preds_seq_span = torch.tensor(preds_seq_span).to(outputs_seq_span.device) # B * S
                preds_seq_attr_start = logits_attr_start.argmax(dim=-1) # B
                if 'StartPrior' in self.model.__class__.__name__:
                    _, _, logits_attr_end  = self.parallel_model(preds_seq_attr_start, *args)
                preds_seq_attr_end = logits_attr_end.argmax(dim=-1) # B

                # get token sequence
                preds_seq_span = preds_seq_span.detach().cpu().numpy()
                preds_seq_attr_start = preds_seq_attr_start.detach().cpu().numpy()
                preds_seq_attr_end = preds_seq_attr_end.detach().cpu().numpy()
                outputs_seq_span = outputs_seq_span.detach().cpu().numpy()
                outputs_seq_attr_start = outputs_seq_attr_start.detach().cpu().numpy()
                outputs_seq_attr_end = outputs_seq_attr_end.detach().cpu().numpy()
                inputs_mask = inputs_mask.detach().cpu().numpy()
                inputs_seq = inputs_seq.detach().cpu().numpy()
                inputs_seq_len = inputs_seq_len.detach().cpu().numpy()
                spos, tpos = 1, -1
                for i in range(bs):
                    seqlen = inputs_seq_len[i]
                    if not self.is_bert_encoder:
                        spos, tpos = 0, seqlen
                    # span metrics update
                    span_batch_metric.update(preds_seq_span[i][:seqlen][spos:tpos], outputs_seq_span[i][:seqlen][spos:tpos])

                    # attr metrics update
                    ## get prediction of sequence
                    preds_seq_attr_id = np.zeros_like(preds_seq_attr_start[i][:seqlen][spos:tpos])
                    preds_seq_attr_id = attr_negid
                    preds_seq_attr_sid = preds_seq_attr_start[i][:seqlen][spos:tpos]
                    preds_seq_attr_eid = preds_seq_attr_end[i][:seqlen][spos:tpos]
                    for i, attr_sid in enumerate(preds_seq_attr_sid):
                        if attr_sid == attr_negid:
                            continue
                        for j, attr_eid in enumerate(preds_seq_attr_eid):
                            if j > 0 and preds_seq_attr_sid[i + j] != attr_negid or j + 1 > 10:
                                break
                            if attr_sid == attr_eid:
                                for k in range(i, i + j + 1):
                                    preds_seq_attr_id[k] = attr_sid
                                break
                    ## get goldlabel of sequence
                    gold_seq_attr_id = np.zeros_like(outputs_seq_attr_start[i][:seqlen][spos:tpos])
                    gold_seq_attr_id = attr_negid
                    gold_seq_attr_sid = outputs_seq_attr_start[i][:seqlen][spos:tpos]
                    gold_seq_attr_eid = outputs_seq_attr_end[i][:seqlen][spos:tpos]
                    for i, attr_sid in enumerate(gold_seq_attr_sid):
                        if attr_sid == attr_negid:
                            continue
                        for j, attr_eid in enumerate(gold_seq_attr_eid):
                            if j > 0 and gold_seq_attr_sid[i + j] != attr_negid or j + 1 > 10:
                                break
                            if attr_sid == attr_eid:
                                for k in range(i, i + j + 1):
                                    gold_seq_attr_id[k] = attr_sid
                                break
                    batch_metric.update(preds_seq_attr_id, gold_seq_attr_id)
                avg_loss.update(loss * bs, bs)
                # log
                if (ith + 1) % 10 == 0:
                    self.logger.info(f'Evaluation...Batches: {ith + 1} finished')

        # get metrics 
        val_loss = avg_loss.avg
        val_span_acc = span_batch_metric.accuracy()
        val_span_prec = span_batch_metric.precision(reduction='micro')
        val_span_rec = span_batch_metric.recall(reduction='micro')
        val_span_f1 = span_batch_metric.f1_score(reduction='micro')
        val_acc = batch_metric.accuracy()
        val_prec = batch_metric.precision(reduction='micro')
        val_rec = batch_metric.recall(reduction='micro')
        val_f1 = batch_metric.f1_score(reduction='micro')
        val_cate_prec = batch_metric.precision(reduction='none')
        val_cate_rec = batch_metric.recall(reduction='none')
        val_cate_f1 = batch_metric.f1_score(reduction='none')
        category_result = {self.model.id2attr[k]: v for k, v in enumerate(zip())}
        result = {'loss': val_loss, 'span_acc': val_span_acc, 'acc': val_acc, 'span_micro_p': val_span_prec, 'span_micro_r': val_span_rec, 'span_micro_f1': val_span_f1, 'micro_p': val_prec, 'micro_r': val_rec, 'micro_f1': val_f1, 'category-p/r/f1': category_result}
        return result
