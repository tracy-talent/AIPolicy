"""
 Author: liujian 
 Date: 2020-10-25 12:38:37 
 Last Modified by: liujian 
 Last Modified time: 2020-10-25 12:38:37 
"""

import sys
sys.path.append('../..')
from pasaie.utils import get_logger
from pasaie.tokenization.utils import load_vocab
from pasaie import pasaner

import torch
import numpy as np
import json
import os
import argparse
import configparser


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-chinese', 
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--bert_name', default='bert', #choices=['bert', 'roberta', 'xlnet', 'albert'], 
        help='bert series model name')
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')
parser.add_argument('--use_lstm', action='store_true', 
        help='whether add lstm encoder on top of bert')
parser.add_argument('--use_crf', action='store_true', 
        help='whether use crf for sequence decode')
parser.add_argument('--tagscheme', default='bio', type=str,
        help='the sequence tag scheme')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
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
parser.add_argument('--batch_size', default=64, type=int,
        help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float,
        help='Learning rate')
parser.add_argument('--bert_lr', default=3e-5, type=float,
        help='Bert Learning rate')
parser.add_argument('--optimizer', default='adamw', type=str,
        help='optimizer:adam|sgd|adamw')
parser.add_argument('--weight_decay', default=1e-5, type=float,
        help='Weight decay')
parser.add_argument('--warmup_step', default=0, type=int,
        help='warmup steps for learning rate scheduler')
parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=3, type=int,
        help='Max number of training epochs')

args = parser.parse_args()

project_path = '/'.join(os.path.abspath(__file__).split('/')[:-3])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))


# construct save path name
def make_dataset_name():
    dataset_name = args.dataset + '_' + args.tagscheme
    return dataset_name
def make_model_name():
    model_name = args.bert_name
    if args.use_lstm:
        model_name += '_lstm'
    if args.use_crf:
        model_name += '_crf'
    return model_name
def make_hparam_string(op, lr, bs, wd, ml):
    return "%s_lr_%.0E,bs=%d,wd=%.0E,ml=%d" % (op, lr, bs, wd, ml)

# logger
os.makedirs(config['path']['ner_log'], exist_ok=True)
logger = get_logger(sys.argv, os.path.abspath(os.path.join(config['path']['ner_log'], f'{make_dataset_name()}_{make_model_name()}.log'))) 

# tensorboard
os.makedirs(config['path']['ner_tb'], exist_ok=True)
tb_logdir = os.path.join(config['path']['ner_tb'], f'{make_dataset_name()}_{make_model_name()}/{make_hparam_string(args.optimizer, args.bert_lr, args.batch_size, args.weight_decay, args.max_length)}')
if os.path.exists(tb_logdir):
    raise Exception(f'path {tb_logdir} exists!')

# Some basic settings
os.makedirs(config['path']['ner_ckpt'], exist_ok=True)
if len(args.ckpt) == 0:
    args.ckpt = f'{make_dataset_name()}_{make_model_name()}'
ckpt = os.path.join(config['path']['ner_ckpt'], f'{args.ckpt}.pth.tar')

if args.dataset != 'none':
    # opennre.download(args.dataset, root_path=root_path)
    args.train_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'train.char.{args.tagscheme}')
    args.val_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'val.char.{args.tagscheme}')
    args.test_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'test.char.{args.tagscheme}')
    args.tag2id_file = os.path.join(config['path']['ner_dataset'], args.dataset, f'tag2id.{args.tagscheme}')
    if not os.path.exists(args.test_file):
        logger.warn("Test file {} does not exist! Use val file instead".format(args.test_file))
        args.test_file = args.val_file
    elif not os.path.exists(args.val_file):
        logger.warn("Val file {} does not exist! Use test file instead".format(args.val_file))
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
sequence_encoder = pasaner.encoder.BERTEncoder(
    max_length=args.max_length,
    pretrain_path=args.pretrain_path,
    bert_name=args.bert_name,
    blank_padding=True
)

# Define the model
model = pasaner.model.BILSTM_CRF(
    sequence_encoder=sequence_encoder, 
    tag2id=tag2id, 
    use_lstm=args.use_lstm, 
    use_crf=args.use_crf
)

# Define the whole training framework
framework = pasaner.framework.Model_CRF(
    model=model,
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
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
    warmup_step=args.warmup_step,
    opt=args.optimizer
)
# framework.load_state_dict(torch.load(ckpt)['state_dict'])

# Train the model
if not args.only_test:
    framework.train_model('micro_f1')

# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
logger.info('Test set results:')
logger.info('Accuracy: {}'.format(result['acc']))
logger.info('Micro precision: {}'.format(result['micro_p']))
logger.info('Micro recall: {}'.format(result['micro_r']))
logger.info('Micro F1: {}'.format(result['micro_f1']))
