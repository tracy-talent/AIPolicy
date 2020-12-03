from ...metrics import Mean

import torch
from torch import nn, optim
import json
from .data_loader import SentenceRELoader, BagRELoader
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

class BagRE(nn.Module):

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
                 opt='sgd',
                 bag_size=0,
                 loss_weight=False):
    
        super().__init__()
        self.max_epoch = max_epoch
        self.bag_size = bag_size
        # Load data
        if train_path != None:
            self.train_loader = BagRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle=True,
                bag_size=bag_size,
                entpair_as_bag=False)

        if val_path != None:
            self.val_loader = BagRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle=False,
                bag_size=bag_size,
                entpair_as_bag=True)
        
        if test_path != None:
            self.test_loader = BagRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle=False,
                bag_size=bag_size,
                entpair_as_bag=True
            )
        # Model
        self.model = nn.DataParallel(model)
        # Criterion
        if loss_weight:
            self.criterion = nn.CrossEntropyLoss(reduction='none', weight=self.train_loader.dataset.weight)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        # Params and optimizer
        params = self.model.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':
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
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'bert_adam'.")
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt
        # logger
        self.logger = logger
        # tensorboard writer
        self.writer = SummaryWriter(tb_logdir)

    def train_model(self, metric='auc'):
        best_metric = 0
        global_step = 0
        negid = -1
        for rel_name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
            if rel_name in self.model.rel2id:
                negid = self.model.rel2id[rel_name]
                break
        for epoch in range(self.max_epoch):
            # Train
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
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                logits = self.model(label, scope, *args, bag_size=self.bag_size)
                loss = self.criterion(logits, label)
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
                if global_step % 10 == 0:
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
                self.optimizer.zero_grad()

            # Val 
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            self.logger.info("AUC: %.4f" % result['pc_auc'])
            self.logger.info("Micro F1: %.4f" % (result['micro_f1']))
            if result[metric] > best_metric:
                self.logger.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'model': self.model.module.state_dict()}, self.ckpt)
                best_metric = result[metric]
            
            # tensorboard val writer
            self.writer.add_scalar('val pc_auc', result['pc_acc'], epoch)
            self.writer.add_scalar('val micro precision', result['micro_p'], epoch)
            self.writer.add_scalar('val micro recall', result['micro_r'], epoch)
            self.writer.add_scalar('val micro f1', result['micro_f1'], epoch)

        self.logger.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.model.eval()
        with torch.no_grad():
            pred_result = []
            for ith, data in enumerate(eval_loader):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                logits = self.model(None, scope, *args, train=False, bag_size=self.bag_size) # results after softmax
                logits = logits.cpu().numpy()
                for i in range(len(logits)):
                    for relid in range(self.model.module.num_class):
                        if self.model.module.id2rel[relid] != 'NA':
                            pred_result.append({
                                'entpair': bag_name[i][:2], 
                                'relation': self.model.module.id2rel[relid], 
                                'score': logits[i][relid]
                            })
            result = eval_loader.dataset.eval(pred_result)
            self.logger.info('Evaluation result: {}.'.format(result))
        return result

    def load_state_dict(self, state_dict):
        self.model.module.load_state_dict(state_dict['model'])
