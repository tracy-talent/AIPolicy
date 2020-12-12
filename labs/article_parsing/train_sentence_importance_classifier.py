# coding:utf-8
import sys
import torch
import json
import os
import re
import argparse
import configparser
import datetime

sys.path.append('../..')
from pasaie.utils import get_logger
from pasaie.utils.sampler import get_relation_sampler
from pasaie.utils.embedding import load_wordvec
import pasaie
from pasaie import pasaap, utils

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='hfl-chinese-bert-wwm-ext',
                    help='Pre-trained ckpt path / model name (hugginface) or pretrained embedding path')
parser.add_argument('--encoder', default='bert',
                    help='encoder name')
parser.add_argument('--model', default='textcnn',
                    help='model name')
parser.add_argument('--ckpt', default='',
                    help='Checkpoint path')
parser.add_argument('--only_test', default=False, type=bool,
                    help='Only run test')
parser.add_argument('--use_sampler', default=False, type=bool,
                    help='Use sampler')
parser.add_argument('--adv', default='none', choices=['fgm', 'pgd', 'flb', 'none'],
                    help='embedding adversarial perturbation')
parser.add_argument('--loss', default='ce', choices=['ce', 'focal', 'dice', 'lsr'],
                    help='loss function')
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                    help='Metric for picking up best checkpoint')
parser.add_argument('--compress_seq', default=False, type=bool,
                    help='whether use pack_padded_sequence to compress mask tokens of batch sequence')

# Dataset
parser.add_argument('--dataset', default='sentence_importance_judgement',
                    help='dataset name in benchmark/article_parsing directory')

# Hyper-parameters
parser.add_argument('--dice_alpha', default=0.6, type=float,
                    help='alpha of dice loss')
parser.add_argument('--dropout_rate', default=0.5, type=float,
                    help='dropout rate')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Learning rate')
parser.add_argument('--bert_lr', default=2e-5, type=float,
                    help='Bert Learning rate')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='optimizer')
parser.add_argument('--max_grad_norm', default=5.0, type=float,
                    help='max_grad_norm for gradient clip')
parser.add_argument('--weight_decay', default=1e-2, type=float,
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
def make_hparam_string(op, lr, bs, wd, ml):
    return "%s_lr_%.0E,bs=%d,wd=%.0E,ml=%d" % (op, lr, bs, wd, ml)


def make_model_name():
    _model_name = '_'.join((args.encoder, args.model, args.loss))
    if len(args.adv) > 0 and args.adv != 'none':
        _model_name += '_' + args.adv
    return _model_name


model_name = make_model_name()

# logger
os.makedirs(os.path.join(config['path']['ap_log'], args.dataset, model_name), exist_ok=True)
logger = get_logger(sys.argv, os.path.join(config['path']['ap_log'], args.dataset, model_name,
                                           f'{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}.log'))

# tensorboard
os.makedirs(config['path']['ap_tb'], exist_ok=True)
tb_logdir = os.path.join(config['path']['ap_tb'], args.dataset, model_name,
                         make_hparam_string(args.optimizer, args.lr, args.batch_size, args.weight_decay,
                                            args.max_length))

# Some basic settings
os.makedirs(config['path']['ap_ckpt'], exist_ok=True)
if len(args.ckpt) == 0:
    args.ckpt = f'{args.dataset}/{model_name}'
ckpt = os.path.join(config['path']['ap_ckpt'], f'{args.ckpt}.pth.tar')

if args.dataset != 'none':
    data_csv_path = os.path.join(config['path']['ap_dataset'], args.dataset, 'full_data.csv')
else:
    raise ValueError('args.dataset cannot be none!')

logger.info('Arguments:')
for arg in vars(args):
    logger.info('{}: {}'.format(arg, getattr(args, arg)))

# Define the sentence encoder
if args.encoder == 'bert':
    sentence_encoder = pasaie.pasaap.encoder.BERTEncoder(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        blank_padding=True
    )
    embedding_dim = 768
elif args.encoder == 'base':
    word2id, word_emb_npy = load_wordvec(wv_file=args.pretrain_path)
    embedding_dim = word_emb_npy.shape[-1]
    sentence_encoder = pasaie.pasaap.encoder.BaseEncoder(token2id=word2id,
                                                         max_length=256,
                                                         embedding_dim=embedding_dim,
                                                         word2vec=word_emb_npy,
                                                         blank_padding=True)
else:
    raise NotImplementedError

# Define the model
if args.model == 'textcnn':
    model = pasaie.pasaap.model.TextCnn(sequence_encoder=sentence_encoder,
                                        num_class=2,
                                        num_filter=256,
                                        embedding_size=embedding_dim,
                                        kernel_sizes=[3, 4, 5],
                                        dropout_rate=args.dropout_rate)
else:
    raise NotImplementedError

if args.use_sampler:
    sampler = None
else:
    sampler = None
# Define the whole training framework
framework = pasaie.pasaap.framework.\
    SentenceImportanceClassifier(model=model,
                                 csv_path=data_csv_path,
                                 ckpt=ckpt,
                                 logger=logger,
                                 tb_logdir=tb_logdir,
                                 compress_seq=args.compress_seq,
                                 batch_size=args.batch_size,
                                 max_epoch=args.max_epoch,
                                 lr=args.lr,
                                 bert_lr=args.bert_lr,
                                 weight_decay=args.weight_decay,
                                 warmup_step=args.warmup_step,
                                 max_grad_norm=args.max_grad_norm,
                                 sampler=sampler,
                                 adv=args.adv,
                                 loss=args.loss,
                                 opt=args.optimizer)
# Train the model
if not args.only_test:
    framework.train_model('micro_f1')

# Test
if args.only_test:
    framework.load_state_dict(torch.load(ckpt))
    framework.eval_model(framework.eval_loader)
