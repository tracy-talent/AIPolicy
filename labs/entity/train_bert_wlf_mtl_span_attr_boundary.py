"""
 Author: liujian
 Date: 2021-01-10 17:36:15
 Last Modified by: liujian
 Last Modified time: 2021-01-10 17:36:15
"""

# coding:utf-8
import sys
sys.path.append('../..')
from pasaie.utils import get_logger, fix_seed
from pasaie.utils.embedding import load_wordvec, construct_embedding_from_numpy
from pasaie.tokenization.utils import load_vocab
from pasaie import pasaner

import torch
import numpy as np
import json
import os
import re
import datetime
import argparse
import logging
import configparser


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-chinese', 
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--bert_name', default='bert', #choices=['bert', 'roberta', 'xlnet', 'albert'], 
        help='bert series model name')
parser.add_argument('--model_type', default='', type=str, choices=['', 'startprior', 'attention', 'mmoe', 'ple', 'plethree', 'pletogether'], 
        help='model type')
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')
parser.add_argument('--share_lstm', action='store_true', 
        help='whether make span and attr share the same lstm after encoder, \
                share_lstm and (span_use_lstm/attr_use_lstm) are mutually exclusive')
parser.add_argument('--span_use_lstm', action='store_true', 
        help='whether use lstm for span sequence after encoder')
parser.add_argument('--attr_use_lstm', action='store_true', 
        help='whether use lstm for attr sequence after encoder')
parser.add_argument('--span_use_crf', action='store_true', 
        help='whether use crf for span sequence decode')
parser.add_argument('--use_mtl_autoweighted_loss', action='store_true', 
        help='whether use automatic weighted loss for multi task learning')
parser.add_argument('--tagscheme', default='bio', type=str,
        help='the sequence tag scheme')
parser.add_argument('--adv', default='', choices=['fgm', 'pgd', 'flb', 'none'],
        help='embedding adversarial perturbation')
parser.add_argument('--loss', default='ce', choices=['ce', 'wce', 'focal', 'dice', 'lsr'],
        help='loss function')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'micro_p', 'micro_r', 'span_acc', 'attr_start_acc', 'attr_end_acc', 'loss'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', choices=['policy', 'weibo', 'resume', 'msra', 'ontonotes4'], 
        help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
        help='Test data file')
parser.add_argument('--span2id_file', default='', type=str,
        help='entity span to ID file')
parser.add_argument('--attr2id_file', default='', type=str,
        help='entity attr to ID file')
parser.add_argument('--char2vec_file', default='', type=str,
        help='character embedding file')
parser.add_argument('--word2vec_file', default='', type=str,
        help='word2vec embedding file')
parser.add_argument('--custom_dict', default='', type=str,
        help='user custom dict for tokenizer toolkit')
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
parser.add_argument('--dropout_rate', default=0.3, type=float,
        help='dropout rate')
parser.add_argument('--optimizer', default='adam', type=str,
        help='optimizer:adam|sgd|adamw')
parser.add_argument('--weight_decay', default=0.05, type=float,
        help='Weight decay')
parser.add_argument('--soft_label', default=False, type=bool, 
        help="whether use one hot for entity span's start label when cat with encoder output")
parser.add_argument('--early_stopping_step', default=3, type=int,
        help='max times of worse metric allowed to avoid overfit')
parser.add_argument('--warmup_step', default=0, type=int,
        help='warmup steps for learning rate scheduler')
parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=3, type=int,
        help='Max number of training epochs')
parser.add_argument('--random_seed', default=12345, type=int,
                    help='global random seed')
parser.add_argument('--experts_layers', default=2, type=int,
                    help='experts layers of PLE MTL')
parser.add_argument('--experts_num', default=2, type=int,
                    help='experts num of every experts in PLE')
args = parser.parse_args()

project_path = '/'.join(os.path.abspath(__file__).split('/')[:-3])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))

#set global random seed
if args.dataset == 'weibo':
   fix_seed(args.random_seed)

# construct save path name
def make_dataset_name():
    dataset_name = args.dataset + '_' + args.tagscheme
    return dataset_name
def make_model_name():
    if args.model_type == 'startprior':
        model_name = 'wlf_mtl_span_attr_boundary_startprior_bert'
    elif args.model_type == 'attention':
        model_name = 'wlf_mtl_span_attr_boundary_attention_bert'
    elif args.model_type == 'mmoe':
        model_name = 'wlf_mtl_span_attr_boundary_mmoe_bert'
    elif args.model_type == 'ple':
        model_name = 'wlf_mtl_span_attr_boundary_ple_bert'
    elif args.model_type == 'plethree':
        model_name = 'wlf_mtl_span_attr_three_boundary_ple_bert'
    elif args.model_type == 'pletogether':
        model_name = 'wlf_mtl_span_attr_boundary_together_ple_bert'
    else:
        model_name = 'wlf_mtl_span_attr_boundary_bert'
    # model_name += '_noact'
    # model_name += '_drop_ln'
    # model_name += '_drop'
    model_name += '_relu_crf1e-2'
    # model_name += '_relu_drop'
    # model_name += '_relu_ln'
    # model_name += '_relu_drop_ln'

    if args.share_lstm:
        model_name += '_sharelstm'
    if args.span_use_lstm:
        model_name += '_spanlstm'
    if args.attr_use_lstm:
        model_name += '_attrlstm'
    if args.span_use_crf:
        model_name += '_spancrf'
    #model_name += '_' + args.optimizer + '_' + str(args.weight_decay) + '_' + args.loss + '_' + str(args.dice_alpha)
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
    args.span2id_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'span2id.{args.tagscheme}')
    if 'together' in args.model_type:
        args.attr2id_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'attr2id_together.{args.tagscheme}')
    else:
        args.attr2id_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'attr2id.{args.tagscheme}')
    if not os.path.exists(args.test_file):
        logger.warn("Test file {} does not exist! Use val file instead".format(args.test_file))
        args.test_file = args.val_file
    elif not os.path.exists(args.val_file):
        logger.warn("Val file {} does not exist! Use test file instead".format(args.val_file))
        args.val_file = args.test_file
else:
    if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.span2id_file) and os.path.exists(args.attr2id_file)):
        raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

logger.info('Arguments:')
for arg in vars(args):
    logger.info('{}: {}'.format(arg, getattr(args, arg)))

#  load tag and vocab
span2id = load_vocab(args.span2id_file)
attr2id = load_vocab(args.attr2id_file)
# load embedding and vocab
word2id, word2vec = load_wordvec(args.word2vec_file)
word2id, word_embedding = construct_embedding_from_numpy(word2id=word2id, word2vec=word2vec)

# Define the sentence encoder
sequence_encoder = pasaner.encoder.BERTWLFEncoder(
    pretrain_path=args.pretrain_path,
    word2id=word2id,
    word_size=word2vec.shape[-1],
    max_length=args.max_length,
    custom_dict=args.custom_dict,
    blank_padding=True
)


# Define the model
if args.model_type == 'attention':
    model = pasaner.model.BILSTM_CRF_Span_Attr_Boundary_Attention(
            sequence_encoder=sequence_encoder, 
            span2id=span2id,
            attr2id=attr2id,
            compress_seq=args.compress_seq,
            share_lstm=args.share_lstm, # False
            span_use_lstm=args.span_use_lstm, # True
            attr_use_lstm=args.attr_use_lstm, # False
            span_use_crf=args.span_use_crf,
            dropout_rate=args.dropout_rate
        )
elif args.model_type == 'startprior':
    model = pasaner.model.BILSTM_CRF_Span_Attr_Boundary_StartPrior(
        sequence_encoder=sequence_encoder, 
        span2id=span2id,
        attr2id=attr2id,
        compress_seq=args.compress_seq,
        share_lstm=args.share_lstm, # False
        span_use_lstm=args.span_use_lstm, # True
        attr_use_lstm=args.attr_use_lstm, # False
        span_use_crf=args.span_use_crf,
        soft_label=args.soft_label,
        dropout_rate=args.dropout_rate
    )
elif args.model_type == 'mmoe':
    model = pasaner.model.BILSTM_CRF_Span_Attr_Boundary_MMoE(
        sequence_encoder=sequence_encoder, 
        span2id=span2id,
        attr2id=attr2id,
        compress_seq=args.compress_seq,
        share_lstm=args.share_lstm, # False
        span_use_lstm=args.span_use_lstm, # True
        attr_use_lstm=args.attr_use_lstm, # False
        span_use_crf=args.span_use_crf,
        dropout_rate=args.dropout_rate
    )
elif args.model_type == 'ple':
    model = pasaner.model.BILSTM_CRF_Span_Attr_Boundary_PLE(
        sequence_encoder=sequence_encoder, 
        span2id=span2id,
        attr2id=attr2id,
        compress_seq=args.compress_seq,
        share_lstm=args.share_lstm, # False
        span_use_lstm=args.span_use_lstm, # True
        attr_use_lstm=args.attr_use_lstm, # False
        span_use_crf=args.span_use_crf,
        dropout_rate=args.dropout_rate,
        experts_layers=args.experts_layers,
        experts_num=args.experts_num
    )
elif args.model_type == 'plethree':
    model = pasaner.model.BILSTM_CRF_Span_Attr_Three_Boundary_PLE(
        sequence_encoder=sequence_encoder, 
        span2id=span2id,
        attr2id=attr2id,
        compress_seq=args.compress_seq,
        share_lstm=args.share_lstm, # False
        span_use_lstm=args.span_use_lstm, # True
        attr_use_lstm=args.attr_use_lstm, # False
        span_use_crf=args.span_use_crf,
        dropout_rate=args.dropout_rate,
        experts_layers=args.experts_layers,
        experts_num=args.experts_num
    )
elif args.model_type == 'pletogether':
    model = pasaner.model.BILSTM_CRF_Span_Attr_Boundary_Together_PLE(
        sequence_encoder=sequence_encoder, 
        span2id=span2id,
        attr2id=attr2id,
        compress_seq=args.compress_seq,
        share_lstm=args.share_lstm, # False
        span_use_lstm=args.span_use_lstm, # True
        attr_use_lstm=args.attr_use_lstm, # False
        span_use_crf=args.span_use_crf,
        dropout_rate=args.dropout_rate,
        experts_layers=args.experts_layers,
        experts_num=args.experts_num
    )
else:
    model = pasaner.model.BILSTM_CRF_Span_Attr_Boundary(
        sequence_encoder=sequence_encoder, 
        span2id=span2id,
        attr2id=attr2id,
        compress_seq=args.compress_seq,
        share_lstm=args.share_lstm, # False
        span_use_lstm=args.span_use_lstm, # True
        attr_use_lstm=args.attr_use_lstm, # False
        span_use_crf=args.span_use_crf,
        dropout_rate=args.dropout_rate
    )

# Define the whole training framework
if 'together' in args.model_type:
    framework_class = pasaner.framework.MTL_Span_Attr_Boundary_Together
else:
    framework_class = pasaner.framework.MTL_Span_Attr_Boundary
framework = framework_class(
    model=model,
    word_embedding=word_embedding,
    train_path=args.train_file if not args.only_test else None,
    val_path=args.val_file if not args.only_test else None,
    test_path=args.test_file if not args.dataset == 'msra' else None,
    ckpt=ckpt,
    logger=logger,
    tb_logdir=tb_logdir,
    compress_seq=args.compress_seq,
    tagscheme=args.tagscheme, 
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    bert_lr=args.bert_lr,
    weight_decay=args.weight_decay,
    early_stopping_step=args.early_stopping_step,
    warmup_step=args.warmup_step, 
    mtl_autoweighted_loss=args.use_mtl_autoweighted_loss,
    opt=args.optimizer,
    loss=args.loss,
    adv=args.adv,
    dice_alpha=args.dice_alpha,
    metric=args.metric,
)

# Load pretrained model
#if ckpt_cnt > 0:
#    logger.info('load checkpoint')
#    framework.load_model(re.sub('\d+\.pth\.tar', f'{ckpt_cnt-1}.pth.tar', ckpt))

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
logger.info('Span Accuracy: {}'.format(result['span_acc']))
if 'together' in args.model_type:
    logger.info('Attr Accuracy: {}'.format(result['attr_acc']))
else:
    logger.info('Attr Start Accuracy: {}'.format(result['attr_start_acc']))
    logger.info('Attr End Accuracy: {}'.format(result['attr_start_acc']))
logger.info('Span Micro precision: {}'.format(result['span_micro_p']))
logger.info('Span Micro recall: {}'.format(result['span_micro_r']))
logger.info('Span Micro F1: {}'.format(result['span_micro_f1']))
logger.info('Micro precision: {}'.format(result['micro_p']))
logger.info('Micro recall: {}'.format(result['micro_r']))
logger.info('Micro F1: {}'.format(result['micro_f1']))
logger.info('Category-P/R/F1: {}'.format(result['category-p/r/f1']))

