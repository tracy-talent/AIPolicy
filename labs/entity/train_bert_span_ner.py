"""
 Author: liujian 
 Date: 2020-10-25 12:38:37 
 Last Modified by: liujian 
 Last Modified time: 2020-10-25 12:38:37 
"""

import sys
sys.path.append('../..')
from pasaie.utils import get_logger, fix_seed
from pasaie.utils.sampler import get_entity_span_single_sampler
from pasaie.tokenization.utils import load_vocab
from pasaie import pasaner

import torch
import numpy as np
import json
import os
import re
import datetime
import argparse
import configparser


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-chinese', 
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--bert_name', default='bert', #choices=['bert', 'roberta', 'xlnet', 'albert'], 
        help='bert series model name')
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--model', default='single', choices=['single', 'multi', 'startprior'], 
        help='used for model choice')
parser.add_argument('--use_sampler', action='store_true',
                    help='Use sampler')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')
parser.add_argument('--use_lstm', action='store_true', 
        help='whether add lstm encoder on top of bert')
parser.add_argument('--use_mtl_autoweighted_loss', action='store_true', 
        help='whether use automatic weighted loss for multi task learning')
parser.add_argument('--tagscheme', default='bio', type=str,
        help='the sequence tag scheme')
parser.add_argument('--adv', default='', choices=['fgm', 'pgd', 'flb', 'none'],
        help='embedding adversarial perturbation')
parser.add_argument('--loss', default='ce', choices=['ce', 'wce', 'focal', 'dice', 'lsr'],
        help='loss function')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'micro_p', 'micro_r', 'acc', 'loss'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', #choices=['policy', 'weibo', 'resume', 'msra', 'ontonotes4'], 
        help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
        help='Test data file')
parser.add_argument('--tag2id_file', default='', type=str,
        help='Relation to ID file')
parser.add_argument('--compress_seq', action='store_true', 
        help='whether use pack_padded_sequence to compress mask tokens of batch sequence')

# Hyper-parameters
parser.add_argument('--dice_alpha', default=0.6, type=float,
        help='alpha of dice loss')
parser.add_argument('--batch_size', default=64, type=int,
        help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float,
        help='Learning rate')
parser.add_argument('--bert_lr', default=3e-5, type=float,
        help='Bert Learning rate')
parser.add_argument('--dropout_rate', default=0.1, type=float,
        help='dropout rate')
parser.add_argument('--ffn_hidden_size', default=150, type=int,
        help='hidden size of FeedForwardNetwork')
parser.add_argument('--width_embedding_size', default=150, type=int,
        help='embedding size of width embedding')
parser.add_argument('--max_span', default=7, type=int, # 最大28，不包括是时间：22，且不包含括号：18,
        help='max length of entity in corpus')
parser.add_argument('--soft_label', default=False, type=bool, 
        help="whether use one hot for entity span's start label when cat with encoder output")
parser.add_argument('--optimizer', default='adamw', type=str,
        help='optimizer:adam|sgd|adamw')
parser.add_argument('--max_grad_norm', default=5.0, type=float,
        help='max_grad_norm for gradient clip')
parser.add_argument('--weight_decay', default=1e-5, type=float,
        help='Weight decay')
parser.add_argument('--early_stopping_step', default=3, type=int,
        help='max times of worse metric allowed to avoid overfit, off when <=0')
parser.add_argument('--warmup_step', default=0, type=int,
        help='warmup steps for learning rate scheduler')
parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=3, type=int,
        help='Max number of training epochs')
parser.add_argument('--random_seed', default=12345, type=int,
                    help='global random seed')

args = parser.parse_args()

project_path = '/'.join(os.path.abspath(__file__).split('/')[:-3])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))

# set global random seed
# fix_seed(args.random_seed)

# construct save path name
def make_dataset_name():
    dataset_name = args.dataset + '_' + args.tagscheme
    return dataset_name
def make_model_name():
    model_name = args.model + '_attr_boundary_' + args.bert_name
    if args.use_lstm:
        model_name += '_lstm'
    model_name += '_' + args.loss
    if 'dice' in args.loss:
        model_name += '_' + str(args.dice_alpha)
    if args.use_mtl_autoweighted_loss:
        model_name += '_autoweighted'
    if len(args.adv) > 0 and args.adv != 'none':
        model_name += '_' + args.adv
    model_name += '_dpr' + str(args.dropout_rate)
    model_name += '_' + args.metric
    return model_name
def make_hparam_string(op, blr, lr, bs, wd, ml):
    return "%s_blr_%.0E_lr_%.0E,bs=%d,wd=%.0E,ml=%d" % (op, blr, lr, bs, wd, ml)
dataset_name = make_dataset_name()
model_name = make_model_name()
hparam_str = make_hparam_string(args.optimizer, args.bert_lr, args.lr, args.batch_size, args.weight_decay, args.max_length)

# logger
os.makedirs(os.path.join(config['path']['ner_log'], dataset_name, model_name), exist_ok=True)
logger = get_logger(sys.argv, os.path.join(config['path']['ner_log'], dataset_name, model_name, 
                    f'{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}.log')) 

# tensorboard
os.makedirs(config['path']['ner_tb'], exist_ok=True)
tb_logdir = os.path.join(config['path']['ner_tb'], dataset_name, model_name, hparam_str)
# if os.path.exists(tb_logdir):
#     raise Exception(f'path {tb_logdir} exists!')

# Some basic settings
os.makedirs(os.path.join(config['path']['ner_ckpt'], dataset_name), exist_ok=True)
if len(args.ckpt) == 0:
    args.ckpt = model_name
ckpt = os.path.join(config['path']['ner_ckpt'], dataset_name, f'{args.ckpt}_0.pth.tar')
ckpt_cnt = 0
while os.path.exists(ckpt):
    ckpt_cnt += 1
    ckpt = re.sub('\d+\.pth\.tar', f'{ckpt_cnt}.pth.tar', ckpt)

if args.dataset != 'none':
    # opennre.download(args.dataset, root_path=root_path)
    if args.dataset == 'msra' or args.dataset == 'ontonotes4':
        args.train_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'train.char.clip256.{args.tagscheme}')
        args.val_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'dev.char.clip256.{args.tagscheme}')
        args.test_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'test.char.clip256.{args.tagscheme}')
    else:
        args.train_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'train.char.{args.tagscheme}')
        args.val_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'dev.char.{args.tagscheme}')
        args.test_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'test.char.{args.tagscheme}')
    args.tag2id_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'attr2id.{args.tagscheme}')
    if not os.path.exists(args.test_file):
        logger.warning("Test file {} does not exist! Use val file instead".format(args.test_file))
        args.test_file = args.val_file
    elif not os.path.exists(args.val_file):
        logger.warning("Val file {} does not exist! Use test file instead".format(args.val_file))
        args.val_file = args.test_file
else:
    if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.tag2id_file)):
        raise Exception('--train_file, --val_file, --test_file and --tag2id_file are not specified or files do not exist. Or specify --dataset')

logger.info('Arguments:')
for arg in vars(args):
    logger.info('{}: {}'.format(arg, getattr(args, arg)))

#  load tag and vocab
tag2id = load_vocab(args.tag2id_file)

# Define the sentence encoder
sequence_encoder = pasaner.encoder.BERT_BILSTM_Encoder(
    max_length=args.max_length,
    pretrain_path=args.pretrain_path,
    bert_name=args.bert_name,
    use_lstm=args.use_lstm if args.model == 'single' else False,
    compress_seq=args.compress_seq if args.model == 'single' else False,
    blank_padding=True
)

# Define the model
if args.model == 'single':
    model = pasaner.model.Span_Cat_CLS(
        sequence_encoder=sequence_encoder, 
        tag2id=tag2id, 
        ffn_hidden_size=args.ffn_hidden_size,
        width_embedding_size=args.width_embedding_size,
        max_span=args.max_span,
        dropout_rate=args.dropout_rate
    )
elif args.model == 'multi':
    model = pasaner.model.Span_Pos_CLS(
        sequence_encoder=sequence_encoder, 
        tag2id=tag2id, 
        use_lstm=args.use_lstm, 
        compress_seq=args.compress_seq, 
        dropout_rate=args.dropout_rate
    )
elif args.model == 'startprior':
    model = pasaner.model.Span_Pos_CLS_StartPrior(
        sequence_encoder=sequence_encoder, 
        tag2id=tag2id, 
        use_lstm=args.use_lstm, 
        soft_label=args.soft_label,
        compress_seq=args.compress_seq, 
        dropout_rate=args.dropout_rate
    )

# Define the whole training framework
if args.use_sampler and args.model == 'single':
    sampler = get_entity_span_single_sampler(args.train_file, tag2id, sequence_encoder, args.max_span, 'WeightedRandomSampler')
else:
    sampler = None
if args.model == 'single':
    framework = pasaner.framework.Span_Single_NER(
        model=model,
        train_path=args.train_file if not args.only_test else None,
        val_path=args.val_file if not args.only_test else None,
        test_path=args.test_file,
        ckpt=ckpt,
        logger=logger,
        tb_logdir=tb_logdir,
        max_span=args.max_span,
        compress_seq=args.compress_seq,
        tagscheme=args.tagscheme,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        bert_lr=args.bert_lr,
        weight_decay=args.weight_decay,
        early_stopping_step=args.early_stopping_step,
        warmup_step=args.warmup_step,
        max_grad_norm=args.max_grad_norm,
        opt=args.optimizer,
        loss=args.loss,
        adv=args.adv,
        dice_alpha=args.dice_alpha,
        metric=args.metric,
        sampler=sampler
    )
elif args.model == 'multi':
    framework = pasaner.framework.Span_Multi_NER(
        model=model,
        train_path=args.train_file if not args.only_test else None,
        val_path=args.val_file if not args.only_test else None,
        test_path=args.test_file if not args.dataset == 'msra' else None,
        ckpt=ckpt,
        logger=logger,
        tb_logdir=tb_logdir,
        max_span=args.max_span,
        compress_seq=args.compress_seq,
        tagscheme=args.tagscheme,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        bert_lr=args.bert_lr,
        weight_decay=args.weight_decay,
        early_stopping_step=args.early_stopping_step,
        warmup_step=args.warmup_step,
        max_grad_norm=args.max_grad_norm,
        mtl_autoweighted_loss=args.use_mtl_autoweighted_loss,
        opt=args.optimizer,
        loss=args.loss,
        adv=args.adv,
        dice_alpha=args.dice_alpha,
        metric=args.metric,
        sampler=sampler
    )
else:
    raise NotImplementedError(f'{args.model} is not implemented.')

# Load pretrained model
# if ckpt_cnt > 0:
#     logger.info('load checkpoint')
#     framework.load_model(re.sub('\d+\.pth\.tar', f'{ckpt_cnt-1}.pth.tar', ckpt))

# Train the model
if not args.only_test:
    framework.train_model()
    framework.load_model(ckpt)

# Test
if 'msra' in args.dataset:
    result = framework.eval_model(framework.val_loader)
else:
    result = framework.eval_model(framework.test_loader)
# Print the result
logger.info('Test set results:')
if args.model == 'single':
    logger.info('Accuracy: {}'.format(result['acc']))
else:
    logger.info('Start Accuracy: {}'.format(result['start_acc']))
    logger.info('End Accuracy: {}'.format(result['end_acc']))
logger.info('Micro precision: {}'.format(result['micro_p']))
logger.info('Micro recall: {}'.format(result['micro_r']))
logger.info('Micro F1: {}'.format(result['micro_f1']))
logger.info('Category-P/R/F1: {}'.format(result['category-p/r/f1']))
