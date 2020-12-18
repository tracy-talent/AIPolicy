"""
 Author: liujian 
 Date: 2020-10-25 14:31:17 
 Last Modified by: liujian 
 Last Modified time: 2020-10-25 14:31:17 
"""

from ...metrics import Mean, micro_p_r_f1_score
from ...utils import extract_kvpairs_in_bio, extract_kvpairs_in_bmoes
# from .data_loader import SingleNERDataLoader
from .data_loader import XLNetSingleNERDataLoader

import os
import datetime
from collections import defaultdict

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter


class XLNet_CRF(nn.Module):
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
                tagscheme='bio', 
                batch_size=32, 
                max_epoch=100, 
                lr=1e-3,
                bert_lr=3e-5,
                weight_decay=1e-5,
                warmup_step=300,
                max_grad_norm=5.0,
                opt='adam'):

        super(XLNet_CRF, self).__init__()
        self.max_epoch = max_epoch
        self.tagscheme = tagscheme
        self.max_grad_norm = max_grad_norm

        # Load Data
        if train_path != None:
            self.train_loader = XLNetSingleNERDataLoader(
                path=train_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=True,
                compress_seq=compress_seq
            )

        if val_path != None:
            self.val_loader = XLNetSingleNERDataLoader(
                path=val_path,
                tag2id=model.tag2id,
                tokenizer=model.sequence_encoder.tokenize,
                batch_size=batch_size,
                shuffle=False,
                compress_seq=compress_seq
            )
        
        if test_path != None:
            self.test_loader = XLNetSingleNERDataLoader(
                path=test_path,
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
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
        # Params and optimizer
        self.lr = lr
        self.bert_lr = bert_lr
        encoder_params = self.parallel_model.module.sequence_encoder.parameters()
        bert_params_id = list(map(id, encoder_params))
        bert_params = list(filter(lambda p: id(p) in bert_params_id, self.parallel_model.parameters()))
        other_params = list(filter(lambda p: id(p) not in bert_params_id, self.parallel_model.parameters()))
        grouped_params = [
            {'params': bert_params, 'lr':bert_lr},
            {'params': other_params, 'lr':lr}
        ]
        if opt == 'sgd':
            self.optimizer = optim.SGD(grouped_params, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(grouped_params) # adam weight_decay is not reasonable
        elif opt == 'adamw': # Optimizer for BERT
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
        self.writer = SummaryWriter(tb_logdir, filename_suffix=datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"))


    def train_model(self, metric='micro_f1'):
        best_metric = 0
        global_step = 0
        negid = -1
        if 'O' in self.model.tag2id:
            negid = self.model.tag2id['O']
        if negid == -1:
            raise Exception("negative tag not is 'O'")

        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
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

                if self.model.crf is None:
                    loss = self.criterion(logits.permute(0, 2, 1), outputs_seq) # B * S
                    loss = torch.sum(loss * inputs_mask, dim=-1) / inputs_seq_len # B
                    preds_seq = logits.argmax(dim=-1) # B * S
                else:
                    log_likelihood = self.model.crf(logits, outputs_seq, mask=inputs_mask, reduction='none')
                    loss = -log_likelihood / inputs_seq_len
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
                for i in range(bs):
                    seqlen = inputs_seq_len[i]
                    spos, tpos = 1, seqlen - 2
                    pred_seq_tag = [self.model.id2tag[tid] for tid in preds_seq[i][-seqlen:][spos:tpos]]
                    gold_seq_tag = [self.model.id2tag[tid] for tid in outputs_seq[i][-seqlen:][spos:tpos]]
                    char_seq = [self.model.sequence_encoder.tokenizer.convert_ids_to_tokens(int(tid)) for tid in inputs_seq[i][-seqlen:][spos:tpos]]
                    # print(char_seq)
                    # print(gold_seq_tag)
                    # print()

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
                avg_loss.update(loss.sum(), bs)
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

                
                # Optimize
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), self.max_grad_norm)
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
                self.save_model(self.ckpt)
                best_metric = result[metric]
            
            # tensorboard val writer
            self.writer.add_scalar('val acc', result['acc'], epoch)
            self.writer.add_scalar('val micro precision', result['micro_p'], epoch)
            self.writer.add_scalar('val micro recall', result['micro_r'], epoch)
            self.writer.add_scalar('val micro f1', result['micro_f1'], epoch)
            
        self.logger.info("Best %s on val set: %f" % (metric, best_metric))


    def eval_model(self, eval_loader):
        self.eval()
        category_result = defaultdict(lambda: [0, 0, 0]) # gold, pred, correct
        preds_kvpairs = []
        golds_kvpairs = []
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
                for i in range(bs):
                    seqlen = inputs_seq_len[i]
                    spos, tpos = 1, seqlen - 2
                    pred_seq_tag = [self.model.id2tag[tid] for tid in preds_seq[i][-seqlen:][spos:tpos]]
                    gold_seq_tag = [self.model.id2tag[tid] for tid in outputs_seq[i][-seqlen:][spos:tpos]]
                    char_seq = [self.model.sequence_encoder.tokenizer.convert_ids_to_tokens(int(tid)) for tid in inputs_seq[i][-seqlen:][spos:tpos]]
                    
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

                # Log
                if (ith + 1) % 20 == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.logger.info(f'Evaluation...Batches: {ith + 1}, acc: {avg_acc.avg:.4f}, micro_p: {prec.avg:.4f}, micro_r: {rec.avg:.4f}, micro_f1: {micro_f1:.4f}')

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
        result = {'acc': avg_acc.avg, 'micro_p': p, 'micro_r':r, 'micro_f1':f1, 'category-p/r/f1':category_result}
        self.logger.info(f'Evaluation result: {result}.')
        return result


    def load_model(self, ckpt):
        state_dict = torch.load(ckpt)
        self.model.load_state_dict(state_dict['model'])
    

    def save_model(self, ckpt):
        state_dict = {'model': self.model.state_dict()}
        torch.save(state_dict, ckpt)
