"""
 Author: liujian
 Date: 2020-12-18 16:02:40
 Last Modified by: liujian
 Last Modified time: 2020-12-18 16:02:40
"""

from ...utils.adversarial import FGM, PGD, FreeLB
from ...losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy

import operator
import datetime

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup


class BaseFramework(nn.Module):
    """if has other torch.nn.Module(exp. AutomaticWeightedLoss) except model, don't extend BaseFramework,
        because it's optimizer will missed other module parameters"""
    
    def __init__(self, 
                model, 
                ckpt, 
                logger, 
                tb_logdir,
                word_embedding=None, 
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
                loss_weight=None,
                dice_alpha=0.6):
        super(BaseFramework, self).__init__()
        encoder_name = model.sequence_encoder.__class__.__name__.lower()
        if 'bert' in encoder_name or 'xlnet' in encoder_name:
            self.is_bert_encoder = True
        else:
            self.is_bert_encoder = False
        self.max_grad_norm = max_grad_norm
        self.max_epoch = max_epoch
        self.metric = metric
        self.early_stopping_step = early_stopping_step
        if word_embedding is not None and word_embedding.weight.requires_grad:
            self.word_embedding = nn.Embedding(*word_embedding.weight.size())
            self.word_embedding.weight.data.copy_(word_embedding.weight.data)
            self.word_embedding.weight.requires_grad = word_embedding.weight.requires_grad
            del word_embedding
        else:
            self.word_embedding = word_embedding

        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(model)
        # Criterion
        if loss == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif loss == 'wce':
            self.criterion = nn.CrossEntropyLoss(weight=loss_weight, reduction='none')
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
        crf_params = [p for n, p in self.named_parameters() if 'crf' in n]
        crf_params_id = [id(p) for p in crf_params] # make crf_params lr=1e-2
        pretrained_params_id = []
        if self.is_bert_encoder:
            encoder_params = self.parallel_model.module.sequence_encoder.parameters()
            pretrained_params_id.extend(list(map(id, encoder_params)))
        pretrained_params = list(filter(lambda p: id(p) in pretrained_params_id, self.parameters()))
        other_params = list(filter(lambda p: id(p) not in pretrained_params_id and id(p) not in crf_params_id, self.parameters()))
        other_params_id = [id(p) for p in other_params]
        grouped_params = [
            {'params': pretrained_params, 'lr': bert_lr},
            {'params': crf_params, 'lr': 1e-2},
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
                    'lr': 1e-2,
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
                    'lr': 1e-2,
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay) and id(p) in other_params_id], 
                    'weight_decay': 0.0,
                    'lr': lr,
                }
            ]
            # adamw_grouped_params = [
            #     {
            #         'params': [p for n, p in params if not any(nd in n for nd in no_decay) and id(p) in bert_params_id], 
            #         'weight_decay': weight_decay,
            #         'lr': bert_lr,
            #     },
            #     {
            #         'params': [p for n, p in params if not any(nd in n for nd in no_decay) and id(p) not in bert_params_id], 
            #         'weight_decay': weight_decay,
            #         'lr': lr,
            #     },
            #     {
            #         'params': [p for n, p in params if any(nd in n for nd in no_decay) and id(p) in bert_params_id], 
            #         'weight_decay': 0.0,
            #         'lr': bert_lr,
            #     },
            #     {
            #         'params': [p for n, p in params if any(nd in n for nd in no_decay) and id(p) not in bert_params_id], 
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
        del self.word_embedding # avoid embedding to cuda
        if torch.cuda.is_available():
            self.cuda()
        self.word_embedding = word_embedding
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
        pass


    def eval_model(self, eval_loader):
        pass


    def load_model(self, ckpt):
        state_dict = torch.load(ckpt)
        self.model.load_state_dict(state_dict['model'])
    
    
    def save_model(self, ckpt):
        state_dict = {'model': self.model.state_dict()}
        torch.save(state_dict, ckpt)
