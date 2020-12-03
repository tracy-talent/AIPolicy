# coding:utf-8
import sys
sys.path.append('../..')
import torch
import numpy as np
import json
import pasare
from pasare import encoder, model, framework
import os
import argparse
import configparser
import logging
from utils import get_logger

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')

# Data
parser.add_argument('--metric', default='auc', choices=['micro_f1', 'auc'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', choices=['none', 'wiki_distant', 'nyt10'],
        help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
        help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
        help='Relation to ID file')

# Bag related
parser.add_argument('--bag_size', type=int, default=0,
        help='Fixed bag size. If set to 0, use original bag sizes')

# Hyper-parameters
parser.add_argument('--batch_size', default=160, type=int,
        help='Batch size')
parser.add_argument('--lr', default=0.1, type=float,
        help='Learning rate')
parser.add_argument('--optimizer', default='adam', type=str,
        help='Optimizer')
parser.add_argument('--weight_decay', default=1e-5, type=float,
        help='Weight decay')
parser.add_argument('--max_length', default=120, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=100, type=int,
        help='Max number of training epochs')

args = parser.parse_args()

project_path = '/'.join(os.path.abspath(__file__).split('/')[:-3])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))

# logger
os.makedirs(config['path']['re_log'], exist_ok=True)
logger = get_logger(sys.argv, os.path.join(config['path']['re_log'], f'{args.dataset}_bag_pcnn_att.log'))

# tensorboard
def make_hparam_string(op, lr, bs, wd, ml):
    return "%s_lr_%.0E,bs=%d,wd=%.0E,ml=%d" % (op, lr, bs, wd, ml)
os.makedirs(config['path']['re_tb'], exist_ok=True)
tb_logdir = os.path.join(config['path']['re_tb'], f'{args.dataset}_bag_pcnn_att/{make_hparam_string(args.optimizer, args.lr, args.batch_size, args.weight_decay, args.max_length)}')
if os.path.exists(tb_logdir):
    raise Exception(f'path {tb_logdir} exists!')

# Some basic settings
os.makedirs(config['path']['re_ckpt'], exist_ok=True)
if len(args.ckpt) == 0:
    args.ckpt = '{}_{}'.format(args.dataset, 'bag_pcnn_att')
ckpt = os.path.join(config['path']['re_ckpt'], f'{args.ckpt}.pth.tar')

if args.dataset != 'none':
    args.train_file = os.path.join(config['path']['re_dataset'], args.dataset, '{}_train.txt'.format(args.dataset))
    args.val_file = os.path.join(config['path']['re_dataset'], args.dataset, '{}_val.txt'.format(args.dataset))
    args.test_file = os.path.join(config['path']['re_dataset'], args.dataset, '{}_test.txt'.format(args.dataset))
    args.rel2id_file = os.path.join(config['path']['re_dataset'], args.dataset, '{}_rel2id.json'.format(args.dataset))
    if not os.path.exists(args.test_file):
        logging.warn("Test file {} does not exist! Use val file instead".format(args.test_file))
        args.test_file = args.val_file
    if args.dataset == 'wiki80' or args.dataset == 'fewrel':
        args.metric = 'acc'
    else:
        args.metric = 'micro_f1'
else:
    if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
        raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))

# Download glove
word2id = json.load(config['path']['glove.6b.50d_word2id.json'])
word2vec = np.load(config['path']['glove.6b.50d_mat.npy'])

# Define the sentence encoder
sentence_encoder = opennre.encoder.PCNNEncoder(
    token2id=word2id,
    max_length=args.max_length,
    word_size=50,
    position_size=5,
    hidden_size=230,
    blank_padding=True,
    kernel_size=3,
    padding_size=1,
    word2vec=word2vec,
    dropout=0.5
)

# Define the model
model = opennre.model.BagAttention(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.BagRE(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    logger=logger,
    tb_logdir=tb_logdir,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    weight_decay=args.weight_decay,
    opt=args.optim,
    bag_size=args.bag_size)

# Train the model
if not args.only_test:
    framework.train_model(args.metric)

# Test the model
framework.load_state_dict(torch.load(ckpt))
result = framework.eval_model(framework.test_loader)

# Print the result
logging.info('Test set results:')
logging.info('AUC: {}'.format(result['auc']))
logging.info('Micro F1: {}'.format(result['micro_f1']))
