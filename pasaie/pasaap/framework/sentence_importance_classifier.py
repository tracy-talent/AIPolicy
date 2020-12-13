import os
import datetime
import time
from collections import defaultdict
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from ...utils.adversarial import FGM, PGD, FreeLB, adversarial_perturbation
from .data_loader import SentenceImportanceDataset, get_train_val_dataloader
from ...metrics import BatchMetric, Mean
from ...losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy


class SentenceImportanceClassifier(nn.Module):
    """model(adaptive) + crf decoder"""

    def __init__(self,
                 model,
                 csv_path,
                 ckpt,
                 logger,
                 tb_logdir,
                 compress_seq=False,
                 batch_size=32,
                 max_epoch=100,
                 lr=1e-3,
                 bert_lr=3e-5,
                 weight_decay=1e-5,
                 warmup_step=300,
                 max_grad_norm=5.0,
                 dice_alpha=0.6,
                 sampler=None,
                 loss='ce',
                 adv='fgm',
                 opt='adam'):

        super(SentenceImportanceClassifier, self).__init__()
        if 'bert' in model.sequence_encoder.__class__.__name__.lower():
            self.is_bert_encoder = True
        else:
            self.is_bert_encoder = False
        self.max_epoch = max_epoch
        self.max_grad_norm = max_grad_norm

        # Load Data
        self.train_loader, self.eval_loader = get_train_val_dataloader(
            csv_path=csv_path,
            sequence_encoder=model.sequence_encoder,
            batch_size=batch_size,
            sampler=sampler,
            compress_seq=compress_seq
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
            self.criterion = DiceLoss(alpha=dice_alpha, gamma=0., reduction='none')
        elif loss == 'lsr':
            self.criterion = LabelSmoothingCrossEntropy(eps=0.1, reduction='none')
        else:
            raise ValueError("Invalid loss. Must be 'ce' or 'focal' or 'dice' or 'lsr'")
        # Params and optimizer
        self.lr = lr
        self.bert_lr = bert_lr
        if self.is_bert_encoder:
            encoder_params = self.parallel_model.module.sequence_encoder.parameters()
            bert_params_id = list(map(id, encoder_params))
        else:
            encoder_params = []
            bert_params_id = []
        bert_params = list(filter(lambda p: id(p) in bert_params_id, self.parallel_model.parameters()))
        other_params = list(filter(lambda p: id(p) not in bert_params_id, self.parallel_model.parameters()))
        grouped_params = [
            {'params': bert_params, 'lr': bert_lr},
            {'params': other_params, 'lr': lr}
        ]
        if opt == 'sgd':
            self.optimizer = optim.SGD(grouped_params, weight_decay=weight_decay, lr=self.lr)
        elif opt == 'adam':
            self.optimizer = optim.Adam(grouped_params)  # adam weight_decay is not reasonable
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
                    'params': [p for n, p in params if
                               not any(nd in n for nd in no_decay) and id(p) not in bert_params_id],
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

    def train_model(self, metric='micro_f1'):
        test_best_metric = 0
        best_metric = 0
        global_step = 0

        for epoch in range(self.max_epoch):
            self.train()
            self.logger.info("=== Epoch %d train ===" % epoch)
            batch_metric = BatchMetric(num_classes=self.model.num_classes)
            avg_loss = Mean()
            t_start = time.time()
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
                torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                # metrics
                batch_metric.update(pred, label, update_score=True)
                avg_loss.update(loss.item() * bs, bs)
                cur_loss = avg_loss.avg
                if self.model.num_classes == 2:
                    cur_acc = cur_prec = cur_recall = cur_f1 = batch_metric.accuracy().item()
                else:
                    cur_acc = batch_metric.accuracy().item()
                    cur_prec = batch_metric.precision().item()
                    cur_recall = batch_metric.recall().item()
                    cur_f1 = batch_metric.f1_score().item()

                # log
                global_step += 1
                if global_step % 20 == 0:
                    self.logger.info(f'Training...Epoches: {epoch}, steps: {global_step}, loss: {cur_loss:.4f},'
                                     f' acc: {cur_acc:.4f}, micro_p: {cur_prec:.4f}, micro_r: {cur_recall:.4f},'
                                     f' micro_f1: {cur_f1:.4f}')

                # tensorboard training writer
                if global_step % 5 == 0:
                    self.writer.add_scalar('train loss', cur_loss, global_step=global_step)
                    self.writer.add_scalar('train acc', cur_acc, global_step=global_step)
                    self.writer.add_scalar('train micro precision', cur_prec, global_step=global_step)
                    self.writer.add_scalar('train micro recall', cur_recall, global_step=global_step)
                    self.writer.add_scalar('train micro f1', cur_f1, global_step=global_step)

            # Val
            self.logger.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.eval_loader)
            self.logger.info('Evaluation result: {}.'.format(result))
            self.logger.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                self.logger.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                # torch.save({'model': self.model.state_dict()}, self.ckpt)
                torch.save(self.model, self.ckpt)
                best_metric = result[metric]

            # tensorboard val writer
            self.writer.add_scalar('val acc', result['acc'], epoch)
            self.writer.add_scalar('val micro precision', result['micro_p'], epoch)
            self.writer.add_scalar('val micro recall', result['micro_r'], epoch)
            self.writer.add_scalar('val micro f1', result['micro_f1'], epoch)
            print(f"Training time for epoch{epoch}: {batch_metric.epoch_time_interval()}s")

            # test
            if hasattr(self, 'test_loader'):
                result = self.eval_model(self.test_loader)
                self.logger.info('Test result: {}.'.format(result))
                self.logger.info('Metric {} current / best: {} / {}'.format(metric, result[metric], test_best_metric))
                if result[metric] > test_best_metric:
                    self.logger.info('Best test ckpt and saved')
                    # torch.save({'model': self.model.state_dict()}, self.ckpt[:-9] + '_test' + self.ckpt[-9:])
                    torch.save(self.model, self.ckpt)
                    test_best_metric = result[metric]

        self.logger.info("Best %s on val set: %f" % (metric, best_metric))
        if hasattr(self, 'test_loader'):
            self.logger.info("Best %s on test set: %f" % (metric, test_best_metric))

    def eval_model(self, eval_loader):
        self.eval()

        batch_metric = BatchMetric(num_classes=self.model.num_classes)
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
                pred = logits.argmax(dim=-1)  # (B)
                batch_metric.update(pred, label, update_score=False)
                # log
                # if (ith + 1) % 20 == 0:
                #     self.logger.info(f'Evaluation...steps: {ith + 1} finished')

        if self.model.num_classes == 2:
            acc = micro_prec = micro_recall = micro_f1 = batch_metric.accuracy().item()
        else:
            acc = batch_metric.accuracy().item()
            micro_prec = batch_metric.precision().item()
            micro_recall = batch_metric.recall().item()
            micro_f1 = batch_metric.f1_score().item()

        cate_prec = batch_metric.precision('none').cpu().tolist()
        cate_rec = batch_metric.recall('none').cpu().tolist()
        cate_f1 = batch_metric.f1_score('none').cpu().tolist()
        category_result = {k: v for k, v in enumerate(zip(cate_prec, cate_rec, cate_f1))}
        result = {'acc': acc, 'micro_p': micro_prec, 'micro_r': micro_recall, 'micro_f1': micro_f1,
                  'category-p/r/f1': category_result}
        return result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
