"""
 Author: liujian
 Date: 2021-02-06 16:13:06
 Last Modified by: liujian
 Last Modified time: 2021-02-06 16:13:06
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
parser.add_argument('--pinyin_embedding_type', default='word_att_add', type=str, choices=['word_att_cat', 'word_att_add', 'char_att_cat', 'char_att_add'],  help='embedding type of pinyin')
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
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
parser.add_argument('--word2vec_file', default='', type=str,
        help='word2vec embedding file')
parser.add_argument('--pinyin2vec_file', default='', type=str,
        help='pinyin2vec embedding file')
parser.add_argument('--word2pinyin_file', default='', type=str,
        help='map from word to pinyin')
parser.add_argument('--custom_dict', default='', type=str,
        help='user custom dict for tokenizer toolkit')  
parser.add_argument('--compress_seq', action='store_true', 
        help='whether use pack_padded_sequence to compress mask tokens of batch sequence')

# Hyper-parameters
parser.add_argument('--dice_alpha', default=0.6, type=float,
        help='alpha of dice loss')
parser.add_argument('--batch_size', default=64, type=int,
        help='Batch size')
parser.add_argument('--crf_lr', default=1e-3, type=float,
        help='CRF Learning rate')
parser.add_argument('--lr', default=1e-3, type=float,
        help='Learning rate')
parser.add_argument('--bert_lr', default=3e-5, type=float,
        help='Bert Learning rate')
parser.add_argument('--dropout_rate', default=0.3, type=float,
        help='dropout rate')
parser.add_argument('--optimizer', default='adam', type=str,
        help='optimizer:adam|sgd|adamw')
parser.add_argument('--max_grad_norm', default=5.0, type=float,
        help='max_grad_norm for gradient clip')
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
parser.add_argument('--max_pinyin_num_of_token', default=10, type=int,
        help='max pinyin num of every token')
parser.add_argument('--max_pinyin_char_length', default=7, type=int,
        help='max length of a pinyin')
parser.add_argument('--lexicon_window_size', default=4, type=int,
        help='upper bound(include) of lexicon window size')
parser.add_argument('--max_epoch', default=3, type=int,
        help='Max number of training epochs')
parser.add_argument('--random_seed', default=12345, type=int,
                    help='global random seed')
parser.add_argument('--experts_layers', default=2, type=int,
                    help='experts layers of PLE MTL')
parser.add_argument('--experts_num', default=2, type=int,
                    help='experts num of every experts in PLE')
parser.add_argument('--group_num', default=3, type=int,
                    help="group by 'bmes' when group_num=4, group by 'bme' when group_num = 3")
parser.add_argument('--pinyin_word_embedding_size', default=50, type=int,
        help='embedding size of pinyin')
parser.add_argument('--pinyin_char_embedding_size', default=50, type=int,
        help='embedding size of pinyin character')
args = parser.parse_args()

project_path = '/'.join(os.path.abspath(__file__).split('/')[:-3])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))

#set global random seed
if args.dataset == 'weibo':
    fix_seed(args.random_seed)

# get lexicon name which used in model_name
if 'sgns_in_ctb' in args.word2vec_file:
    lexicon_name = 'sgns_in_ctb'
elif 'tencent_in_ctb' in args.word2vec_file:
    lexicon_name = 'tencent_in_ctb'
elif 'ctb' in args.word2vec_file:
    lexicon_name = 'ctb'
elif 'sgns' in args.word2vec_file:
    lexicon_name = 'sgns'
elif 'giga' in args.word2vec_file:
    lexicon_name = 'giga'
elif 'tencent' in args.word2vec_file:
    lexicon_name = 'tencent'
else:
    raise FileNotFoundError(f'{args.word2vec_file} is not found!')
# construct save path name
def make_dataset_name():
    dataset_name = args.dataset + '_' + args.tagscheme
    return dataset_name
def make_model_name():
    model_name = f'bmes{args.group_num}_lexicon_{lexicon_name}_window{args.lexicon_window_size}_pinyin_{args.pinyin_embedding_type}_mtl_attr_boundary_bert'
    if args.use_lstm:
        model_name += '_lstm'
    model_name += '_' + args.loss
    if 'dice' in args.loss:
        model_name += str(args.dice_alpha)
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
tag2id = load_vocab(args.tag2id_file)
# load word embedding and vocab
word2id, word2vec = load_wordvec(args.word2vec_file, binary='.bin' in args.word2vec_file)
word2id, word_embedding = construct_embedding_from_numpy(word2id=word2id, word2vec=word2vec, finetune=False)
# load pinyin embedding and vocab
pinyin2id, pinyin2vec = load_wordvec(args.pinyin2vec_file, binary='.bin' in args.pinyin2vec_file)
pinyin2id, pinyin_embedding = construct_embedding_from_numpy(word2id=pinyin2id, word2vec=pinyin2vec, finetune=False)
# load map from word to pinyin
if 'char' in args.pinyin_embedding_type:
    pinyin_char2id = {'[PAD]': 0, '[UNK]': 1, '\'': 2}
    pinyin2id = {'[PAD]': 0, '[UNK]': 1}
    pinyin_num = len(pinyin2id)
    pinyin_char_num = len(pinyin_char2id)
    word2pinyin = {}
    with open(args.word2pinyin_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            line[1] = eval(line[1])
            word2pinyin[line[0]] = line[1]
            for p in line[1]:
                if p not in pinyin2id:
                    pinyin2id[p] = pinyin_num
                    pinyin_num += 1
                for c in p:
                    if c not in pinyin_char2id:
                        pinyin_char2id[c] = pinyin_char_num
                        pinyin_char_num += 1

# Define the sentence encoder
if args.pinyin_embedding_type == 'word_att_cat':
    sequence_encoder = pasaner.encoder.BERT_BMES_Lexicon_PinYin_Word_Attention_Cat_Encoder(
        pretrain_path=args.pretrain_path,
        word2id=word2id,
        pinyin2id=pinyin2id,
        pinyin_embedding=pinyin_embedding,
        word_size=word2vec.shape[-1],
        lexicon_window_size=args.lexicon_window_size,
        pinyin_size=pinyin2vec.shape[-1],
        max_length=args.max_length,
        group_num=args.group_num,
        blank_padding=True
    )
elif args.pinyin_embedding_type == 'word_att_add':
    sequence_encoder = pasaner.encoder.BERT_BMES_Lexicon_PinYin_Word_Attention_Add_Encoder(
        pretrain_path=args.pretrain_path,
        word2id=word2id,
        pinyin2id=pinyin2id,
        pinyin_embedding=pinyin_embedding,
        word_size=word2vec.shape[-1],
        lexicon_window_size=args.lexicon_window_size,
        pinyin_size=pinyin2vec.shape[-1],
        max_length=args.max_length,
        group_num=args.group_num,
        blank_padding=True
    )
elif args.pinyin_embedding_type == 'char_att_cat':
    sequence_encoder = pasaner.encoder.BERT_BMES_Lexicon_PinYin_Char_Attention_Cat_Encoder(
        pretrain_path=args.pretrain_path,
        word2id=word2id,
        pinyin_char2id=pinyin_char2id,
        word_size=word2vec.shape[-1],
        lexicon_window_size=args.lexicon_window_size,
        pinyin_char_size=args.pinyin_char_embedding_size,
        max_pinyin_char_length=args.max_pinyin_char_length,
        max_length=args.max_length,
        group_num=args.group_num,
        blank_padding=True
    )
elif args.pinyin_embedding_type == 'char_att_add':
    sequence_encoder = pasaner.encoder.BERT_BMES_Lexicon_PinYin_Char_Attention_Add_Encoder(
        pretrain_path=args.pretrain_path,
        word2id=word2id,
        pinyin_char2id=pinyin_char2id,
        word_size=word2vec.shape[-1],
        lexicon_window_size=args.lexicon_window_size,
        pinyin_char_size=args.pinyin_char_embedding_size,
        max_pinyin_char_length=args.max_pinyin_char_length,
        max_length=args.max_length,
        group_num=args.group_num,
        blank_padding=True
    )
else:
    raise NotImplementedError(f'args.pinyin_embedding_type: {args.pinyin_embedding_type} is not supported by exsited model currently.')

# Define the model
model = pasaner.model.Span_Pos_CLS(
        sequence_encoder=sequence_encoder, 
        tag2id=tag2id, 
        use_lstm=args.use_lstm, 
        compress_seq=args.compress_seq, 
        dropout_rate=args.dropout_rate
    )

# Define the whole training framework
framework = pasaner.framework.Span_Multi_NER(
        model=model,
        train_path=args.train_file if not args.only_test else None,
        val_path=args.val_file if not args.only_test else None,
        test_path=args.test_file if not args.dataset == 'msra' else None,
        ckpt=ckpt,
        logger=logger,
        tb_logdir=tb_logdir,
        word_embedding=word_embedding,
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
        max_grad_norm=args.max_grad_norm,
        opt=args.optimizer,
        loss=args.loss,
        adv=args.adv,
        dice_alpha=args.dice_alpha,
        metric=args.metric
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
logger.info('Start Accuracy: {}'.format(result['start_acc']))
logger.info('End Accuracy: {}'.format(result['end_acc']))
logger.info('Micro precision: {}'.format(result['micro_p']))
logger.info('Micro recall: {}'.format(result['micro_r']))
logger.info('(w{:d}, dpr{:.1f})Micro F1: {}'.format(args.lexicon_window_size, args.dropout_rate, result['micro_f1']))
logger.info('Category-P/R/F1: {}'.format(result['category-p/r/f1']))

