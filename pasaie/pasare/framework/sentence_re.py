from ...metrics import Mean
from ...losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy
from ...utils.adversarial import FGM, PGD, FreeLB, adversarial_perturbation
from .data_loader import SentenceRELoader

import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import datetime


class SentenceRE(nn.Module):

    def __init__(self,
                 model,
                 train_path,
                 val_path,
                 test_path,
                 ckpt,
                 logger,
                 tb_logdir,
                 compress_seq=True,
                 batch_size=32,
                 max_epoch=100,
                 lr=1e-3,
                 bert_lr=2e-5,
                 weight_decay=1e-5,
                 warmup_step=300,
                 max_grad_norm=5.0,
                 adv='fgm',
                 loss='ce',
                 opt='sgd',
                 sampler=None):

        super().__init__()
        if 'bert' in model.sentence_encoder.__class__.__name__.lower():
            self.is_bert_encoder = True
        else:
            self.is_bert_encoder = False
        self.max_epoch = max_epoch
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
            self.criterion = DiceLoss(alpha=1., gamma=0., reduction='none')
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
            self.optimizer = optim.SGD(grouped_params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(grouped_params, lr, weight_decay=weight_decay)
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
            # grouped_params = [
            #     {
            #         'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
            #         'weight_decay': weight_decay,
            #         'lr': lr,
            #         'ori_lr': lr
            #     },
            #     {
            #         'params': [p for n, p in params if any(nd in n for nd in no_decay)],
            #         'weight_decay': 0.0,
            #         'lr': lr,
            #         'ori_lr': lr
            #     }
            # ]
            self.optimizer = AdamW(adamw_grouped_params, correct_bias=True)  # original: correct_bias=False
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = None
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
        

    def train_model(self, metric='acc'):
        best_metric = 0
        global_step = 0
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
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                # metrics
                acc = (pred == label).long().sum().item()
                label_pos = (label != negid).long().sum().item()
                pred_pos = (pred != negid).long().sum().item()
                true_pos = ((pred == label).long() * (label != negid).long()).sum().item()
                avg_loss.update(loss.item() * bs, bs)
                avg_acc.update(acc, bs)
                prec.update(true_pos, pred_pos)
                rec.update(true_pos, label_pos)

                # log
                global_step += 1
                if global_step % 20 == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.logger.info(
                        f'Training...Epoches: {epoch}, steps: {global_step}, loss: {avg_loss.avg:.4f}, acc: {avg_acc.avg:.4f}, micro_p: {prec.avg:.4f}, micro_r: {rec.avg:.4f}, micro_f1: {micro_f1:.4f}')

                # tensorboard training writer
                if global_step % 5 == 0:
                    micro_f1 = 2 * prec.avg * rec.avg / (prec.avg + rec.avg) if (prec.avg + rec.avg) > 0 else 0
                    self.writer.add_scalar('train loss', avg_loss.avg, global_step=global_step)
                    self.writer.add_scalar('train acc', avg_acc.avg, global_step=global_step)
                    self.writer.add_scalar('train micro precision', prec.avg, global_step=global_step)
                    self.writer.add_scalar('train micro recall', rec.avg, global_step=global_step)
                    self.writer.add_scalar('train micro f1', micro_f1, global_step=global_step)

            # Val 
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            self.logger.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                self.logger.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]

            # tensorboard val writer
            self.writer.add_scalar('val acc', result['acc'], epoch)
            self.writer.add_scalar('val micro precision', result['micro_p'], epoch)
            self.writer.add_scalar('val micro recall', result['micro_r'], epoch)
            self.writer.add_scalar('val micro f1', result['micro_f1'], epoch)

        self.logger.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = Mean()
        pred_result = []
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
                score, pred = logits.max(-1)  # (B)
                # Save result
                pred_result.extend(pred.tolist())
                # Log
                acc = (pred == label).long().sum().item()
                avg_acc.update(acc, label.size(0))
                if (ith + 1) % 20 == 0:
                    self.logger.info(f'Evaluation...Batches: {ith + 1}, val_acc: {avg_acc.avg:.4f}')
        result = eval_loader.dataset.eval(pred_result)
        self.logger.info('Evaluation result: {}.'.format(result))
        return result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
