from ...metrics import Mean

import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from torch.utils.tensorboard import SummaryWriter

class SentenceRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path, 
                 test_path,
                 ckpt,
                 logger, 
                 tb_logdir,
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 warmup_step=300,
                 opt='sgd'):
    
        super().__init__()
        self.max_epoch = max_epoch
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle=True)

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle=False)
        
        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle=False)
                
        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw': # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=True) # original: correct_bias=False
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
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
                loss = self.criterion(logits, label) # B
                score, pred = logits.max(-1) # (B)
                acc = (pred == label).long().sum().item()
                label_pos = (label != negid).long().sum().item()
                pred_pos = (pred != negid).long().sum().item()
                true_pos = ((pred == label).long() * (label != negid).long()).sum().item()

                # Log
                avg_loss.update(loss.sum().item(), label.size(0))
                avg_acc.update(acc, label.size(0))
                prec.update(true_pos, pred_pos)
                rec.update(true_pos, label_pos)
                global_step += 1
                if global_step % 20 == 0:
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
                score, pred = logits.max(-1) # (B)
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

