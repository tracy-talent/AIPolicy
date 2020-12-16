# coding:utf-8
import sys
import torch
import json
import os
import re
import argparse
import configparser
import datetime
from ast import literal_eval

sys.path.append('../..')
from pasaie.utils import get_logger
from pasaie.utils.embedding import load_wordvec
import pasaie
from pasaie import pasaap, utils

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='hfl-chinese-bert-wwm-ext',
                    help='Pre-trained ckpt path / model name (hugginface) or pretrained embedding path')
parser.add_argument('--encoder', default='bert', choices=['bert', 'base'], 
                    help='encoder name')
parser.add_argument('--bert_name', default='bert', choices=['bert', 'roberta', 'albert'], 
        help='bert series model name')
parser.add_argument('--model', default='textcnn',
                    help='model name')
parser.add_argument('--ckpt', default='',
                    help='Checkpoint path')
parser.add_argument('--only_test', action='store_true',
                    help='Only run test')
parser.add_argument('--use_sampler', action='store_true',
                    help='Use sampler')
parser.add_argument('--adv', default='none', choices=['fgm', 'pgd', 'flb', 'none'],
                    help='embedding adversarial perturbation')
parser.add_argument('--loss', default='bce', choices=['pwbce', 'bce', 'ce', 'wce', 'focal', 'dice', 'lsr'],
                    help='loss function')
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                    help='Metric for picking up best checkpoint')
parser.add_argument('--compress_seq', action='store_true',
                    help='whether use pack_padded_sequence to compress mask tokens of batch sequence')
parser.add_argument('--neg_classes', default='', type=str,
                    help='list of negtive classes id')

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
    if args.encoder == 'bert':
        _model_name = '_'.join((args.bert_name, args.model, args.loss))
    else:
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

ckpt = os.path.join(config['path']['ap_ckpt'], f'{args.ckpt}0.pth.tar')
ckpt_cnt = 0
while os.path.exists(ckpt):
    ckpt_cnt += 1
    ckpt = re.sub('\d+\.pth\.tar', f'{ckpt_cnt}.pth.tar', ckpt)

if args.dataset != 'none':
    data_csv_path = os.path.join(config['path']['ap_dataset'], args.dataset, 'test_data.csv')
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
        bert_name=args.bert_name,
        blank_padding=True
    )
elif args.encoder == 'base':
    word2id, word_emb_npy = load_wordvec(wv_file=args.pretrain_path)
    sentence_encoder = pasaie.pasaap.encoder.BaseEncoder(token2id=word2id,
                                                         max_length=256,
                                                         word_embe_size=word_emb_npy.shape[-1],
                                                         word2vec=word_emb_npy,
                                                         blank_padding=True)
else:
    raise NotImplementedError

# Define the model
if args.model == 'textcnn':
    model = pasaie.pasaap.model.TextCnn(sequence_encoder=sentence_encoder,
                                        num_class=1 if 'bce' in args.loss else 2,
                                        num_filter=256,
                                        kernel_sizes=[3, 4, 5],
                                        dropout_rate=args.dropout_rate)
elif args.model == 'bilstm':
    model = pasaie.pasaap.model.BilstmAttn(
        sequence_encoder=sentence_encoder,
        num_class=1 if 'bce' in args.loss else 2,
        hidden_size=128,
        num_layers=1,
        num_heads=8,
        dropout_rate=0.2,
        compress_seq=args.compress_seq,
        use_attn=True
    )
else:
    raise NotImplementedError

if args.use_sampler:
    sampler = None
else:
    sampler = 'WeightedRandomSampler'

if args.neg_classes:
    args.neg_classes = literal_eval(args.neg_classes)
else:
    args.neg_classes = []
# Define the whole training framework
framework = pasaie.pasaap.framework.\
    SentenceImportanceClassifier(model=model,
                                 csv_path=data_csv_path,
                                 neg_classes=args.neg_classes,
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

# Load pretrained model
if ckpt_cnt > 0:
    framework.load_state_dict(torch.load(re.sub('\d+\.pth\.tar', f'{ckpt_cnt-1}.pth.tar', ckpt)))

# Train the model
if not args.only_test:
    framework.train_model('micro_f1')

# Test
if not args.only_test:
    framework.load_state_dict(torch.load(ckpt))
framework.eval_model(framework.eval_loader)

# Print the result
logger.info('Test set best results:')
logger.info('Accuracy: {}'.format(result['acc']))
logger.info('Micro precision: {}'.format(result['micro_p']))
logger.info('Micro recall: {}'.format(result['micro_r']))
logger.info('Micro F1: {}'.format(result['micro_f1']))
