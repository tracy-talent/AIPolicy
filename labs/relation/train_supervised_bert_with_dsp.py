"""
 Author: liujian
 Date: 2020-12-17 22:14:32
 Last Modified by: liujian
 Last Modified time: 2020-12-17 22:14:32
"""

# coding:utf-8
import sys
sys.path.append('../..')
from pasaie.utils import get_logger, fix_seed
from pasaie.utils.sampler import get_relation_sampler
from pasaie import pasare

import torch
import json
import os
import re
import datetime
import argparse
import configparser
from ast import literal_eval


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased',
                    help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--language', default='en', choices=['en', 'zh'], 
                    help='laguage of bert available to')
parser.add_argument('--bert_name', default='bert', choices=['bert', 'roberta', 'albert'], 
        help='bert series model name')
parser.add_argument('--ckpt', default='',
                    help='Checkpoint name')
parser.add_argument('--encoder_type', default='entity', choices=['dsp', 'entity_dsp', 'entity_context_dsp'],
                    help='Sentence representation model type')
parser.add_argument('--only_test', action='store_true',
                    help='Only run test')
parser.add_argument('--mask_entity', action='store_true',
                    help='Mask entity mentions')
parser.add_argument('--use_attention4context', action='store_true',
                    help='whether use attention for DSP feature, otherwise use conv')
parser.add_argument('--use_attention4dsp', action='store_true',
                    help='whether use attention for DSP feature')
parser.add_argument('--embed_entity_type', action='store_true',
                    help='Embed entity-type information in RE training process')
parser.add_argument('--adv', default='', choices=['fgm', 'pgd', 'flb', 'none'],
        help='embedding adversarial perturbation')
parser.add_argument('--loss', default='ce', choices=['ce', 'focal', 'dice', 'lsr'],
        help='loss function')
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'micro_p', 'micro_r', 'acc', 'loss'],
                    help='Metric for picking up best checkpoint')
parser.add_argument('--dsp_tool', default='ddp', choices=['ltp', 'ddp', 'stanza'],
        help='DSP tool used')

# Data
parser.add_argument('--dataset', default='none',
                    choices=['none', 'semeval', 'kbp37', 'wiki80', 'tacred', 'policy', 'nyt10', 'test-policy'],
                    help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
                    help='Training data file')
parser.add_argument('--val_file', default='', type=str,
                    help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
                    help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
                    help='Relation to ID file')
parser.add_argument('--compress_seq', action='store_true',
                    help='whether use pack_padded_sequence to compress mask tokens of batch sequence')
parser.add_argument('--dsp_preprocessed', action='store_true',
                    help='whether have preprocessed dsp path of head anf tail entity to root')
parser.add_argument('--neg_classes', default='', type=str,
                    help='list of negtive classes id')
parser.add_argument('--use_sampler', action='store_true',
                    help='Use sampler')

# Hyper-parameters
parser.add_argument('--dice_alpha', default=0.6, type=float,
        help='alpha of dice loss')
parser.add_argument('--dropout_rate', default=0.1, type=float,
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
parser.add_argument('--early_stopping_step', default=3, type=int,
                    help='max times of worse metric allowed to avoid overfit, off when <=0')
parser.add_argument('--warmup_step', default=0, type=int,
                    help='warmup steps for learning rate scheduler')
parser.add_argument('--max_length', default=256, type=int,
                    help='Maximum sentence length')
parser.add_argument('--max_dsp_path_length', default=15, type=int,
                    help='Maximum entity to root dsp path length') # true max length {ltp:9, ddp:12, stanza:12}, suggest 15 for ddp/stanza, 10 for ltp
parser.add_argument('--max_epoch', default=3, type=int,
                    help='Max number of training epochs')
parser.add_argument('--random_seed', default=12345, type=int,
                    help='global random seed')

args = parser.parse_args()

project_path = '/'.join(os.path.abspath(__file__).split('/')[:-3])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))

# set global random seed
fix_seed(args.random_seed)

# construct save path name
def make_model_name():
    model_name = args.bert_name + '_' + args.encoder_type
    if args.embed_entity_type:
        model_name += '_embed_entity'
    model_name += '_tail_bert_' + args.dsp_tool + '_dsp'
    if args.use_attention4context:
        model_name += '_attention_context'
    elif 'context' in args.encoder_type:
        model_name += '_conv_context'
    model_name += '_' + args.loss
    if 'dice' in args.loss:
        model_name += str(args.dice_alpha)
    if len(args.adv) > 0 and args.adv != 'none':
        model_name += '_' + args.adv
    model_name += '_dpr' + str(args.dropout_rate)
    model_name += '_' + args.metric
    return model_name
def make_hparam_string(op, blr, lr, bs, wd, ml):
    return "%s_blr_%.0E_lr_%.0E,bs=%d,wd=%.0E,ml=%d" % (op, blr, lr, bs, wd, ml)
model_name = make_model_name()
hparam_str = make_hparam_string(args.optimizer, args.bert_lr, args.lr, args.batch_size, args.weight_decay, args.max_length)

# logger
os.makedirs(os.path.join(config['path']['re_log'], args.dataset, model_name), exist_ok=True)
logger = get_logger(sys.argv, os.path.join(config['path']['re_log'], args.dataset, model_name, 
                                        f'{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}.log'))

# tensorboard
os.makedirs(config['path']['re_tb'], exist_ok=True)
tb_logdir = os.path.join(config['path']['re_tb'], args.dataset, model_name, hparam_str)

# Some basic settings
os.makedirs(os.path.join(config['path']['re_ckpt'], args.dataset), exist_ok=True)
if len(args.ckpt) == 0:
    args.ckpt = model_name
ckpt = os.path.join(config['path']['re_ckpt'], args.dataset, f'{args.ckpt}_0.pth.tar')
ckpt_cnt = 0
while os.path.exists(ckpt):
    ckpt_cnt += 1
    ckpt = re.sub('\d+\.pth\.tar', f'{ckpt_cnt}.pth.tar', ckpt)

if args.dataset != 'none':
    args.train_file = os.path.join(config['path']['re_dataset'], args.dataset, '{}_train.txt'.format(args.dataset))
    args.val_file = os.path.join(config['path']['re_dataset'], args.dataset, '{}_val.txt'.format(args.dataset))
    args.test_file = os.path.join(config['path']['re_dataset'], args.dataset, '{}_test.txt'.format(args.dataset))
    args.rel2id_file = os.path.join(config['path']['re_dataset'], args.dataset, '{}_rel2id.json'.format(args.dataset))
    if args.embed_entity_type:
        args.tag2id_file = os.path.join(config['path']['re_dataset'], args.dataset,
                                        '{}_tag2id.json'.format(args.dataset))
    if not os.path.exists(args.test_file):
        logger.warning("Test file {} does not exist! Use val file instead".format(args.test_file))
        args.test_file = args.val_file
    # if args.dataset == 'wiki80' or args.dataset == 'fewrel':
    #     args.metric = 'acc'
    # else:
    #     args.metric = 'micro_f1'
else:
    if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(
            args.test_file) and os.path.exists(args.rel2id_file)):
        raise Exception(
            f'--train_file, --val_file, --test_file and --rel2id_file are not specified for dataset `{args.dataset}` or files do not exist. Or specify --dataset')

logger.info('Arguments:')
for arg in vars(args):
    logger.info('{}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))
tag2id = None if not args.embed_entity_type else json.load(open(args.tag2id_file))

# Define the sentence encoder
if args.encoder_type == 'entity_dsp':
    sentence_encoder = pasare.encoder.BERTEntityWithDSPEncoder(
        pretrain_path=args.pretrain_path,
        bert_name=args.bert_name,
        max_length=args.max_length,
        max_dsp_path_length=args.max_dsp_path_length if not args.dsp_preprocessed else -1,
        dsp_tool=args.dsp_tool,
        tag2id=tag2id,
        use_attention4dsp=args.use_attention4dsp,
        mask_entity=args.mask_entity,
        blank_padding=True,
        compress_seq=args.compress_seq,
        language=args.language
    )
elif args.encoder_type == 'entity_context_dsp':
    sentence_encoder = pasare.encoder.BERTEntityWithContextDSPEncoder(
        pretrain_path=args.pretrain_path,
        bert_name=args.bert_name,
        max_length=args.max_length,
        max_dsp_path_length=args.max_dsp_path_length if not args.dsp_preprocessed else -1,
        dsp_tool=args.dsp_tool,
        tag2id=tag2id,
        use_attention4context=args.use_attention4context,
        use_attention4dsp=args.use_attention4dsp,
        mask_entity=args.mask_entity,
        blank_padding=True,
        compress_seq=args.compress_seq,
        language=args.language
    )
elif args.encoder_type == 'dsp':
    sentence_encoder = pasare.encoder.BERTWithDSPEncoder(
        pretrain_path=args.pretrain_path,
        bert_name=args.bert_name,
        max_length=args.max_length,
        max_dsp_path_length=args.max_dsp_path_length if not args.dsp_preprocessed else -1,
        dsp_tool=args.dsp_tool,
        use_attention4dsp=args.use_attention4dsp,
        mask_entity=args.mask_entity,
        blank_padding=True,
        compress_seq=args.compress_seq,
        language=args.language
    )
else:
    raise NotImplementedError

# Define the model
model = pasare.model.SoftmaxNN(
    sentence_encoder=sentence_encoder, 
    num_class=len(rel2id), 
    rel2id=rel2id, 
    dropout_rate=args.dropout_rate)

# Define the whole training framework
if args.neg_classes:
    args.neg_classes = literal_eval(args.neg_classes)
else:
    args.neg_classes = []
if args.use_sampler:
    sampler = get_relation_sampler(args.train_file, rel2id, 'WeightedRandomSampler')
else:
    sampler = None
if args.dsp_preprocessed:
    if 'multilingual' in args.pretrain_path:
        dsp_file_path_suffix = f'_tail_{args.bert_name}_multilingual_{args.dsp_tool}_dsp_path.txt'
    elif 'large-uncased-wwm' in args.pretrain_path:
        dsp_file_path_suffix = f'_tail_{args.bert_name}_large_uncased_wwm_{args.dsp_tool}_dsp_path.txt'
    elif 'large-uncased' in args.pretrain_path:
        dsp_file_path_suffix = f'_tail_{args.bert_name}_large_uncased_{args.dsp_tool}_dsp_path.txt'
    elif 'base-uncased' in args.pretrain_path:
        dsp_file_path_suffix = f'_tail_{args.bert_name}_base_uncased_{args.dsp_tool}_dsp_path.txt'
    elif 'uncased' in args.pretrain_path:
        dsp_file_path_suffix = f'_tail_{args.bert_name}_uncased_{args.dsp_tool}_dsp_path.txt'
    elif 'large-cased-wwm' in args.pretrain_path:
        dsp_file_path_suffix = f'_tail_{args.bert_name}_large_cased_wwm_{args.dsp_tool}_dsp_path.txt'
    elif 'large-cased' in args.pretrain_path:
        dsp_file_path_suffix = f'_tail_{args.bert_name}_large_cased_{args.dsp_tool}_dsp_path.txt'
    elif 'base-cased' in args.pretrain_path:
        dsp_file_path_suffix = f'_tail_{args.bert_name}_base_cased_{args.dsp_tool}_dsp_path.txt'
    elif 'cased' in args.pretrain_path:
        dsp_file_path_suffix = f'_tail_{args.bert_name}_cased_{args.dsp_tool}_dsp_path.txt'
    else:
        dsp_file_path_suffix = f'_tail_{args.bert_name}_{args.dsp_tool}_dsp_path.txt'

framework = pasare.framework.SentenceWithDSPRE(
    train_path=args.train_file if not args.only_test else None,
    val_path=args.val_file if not args.only_test else None,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    logger=logger,
    tb_logdir=tb_logdir,
    neg_classes=args.neg_classes,
    compress_seq=args.compress_seq,
    max_dsp_path_length=args.max_dsp_path_length if args.dsp_preprocessed else -1,
    dsp_file_path_suffix=dsp_file_path_suffix if args.dsp_preprocessed else None,
    dsp_tool=args.dsp_tool,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    bert_lr=args.bert_lr,
    weight_decay=args.weight_decay,
    early_stopping_step=args.early_stopping_step,
    warmup_step=args.warmup_step,
    max_grad_norm=args.max_grad_norm,
    dice_alpha=args.dice_alpha,
    metric=args.metric,
    adv=args.adv,
    loss=args.loss,
    opt=args.optimizer,
    sampler=sampler
)

# Load pretrained model
# if ckpt_cnt > 0:
#     logger.info('load checkpoint')
#     framework.load_model(re.sub('\d+\.pth\.tar', f'{ckpt_cnt-1}.pth.tar', ckpt))

# Train the model
if not args.only_test:
    framework.train_model()
    framework.load_model(ckpt)

# Test
result = framework.eval_model(framework.test_loader)
# Print the result
logger.info('Test set best results:')
logger.info('Accuracy: {}'.format(result['acc']))
logger.info('Micro precision: {}'.format(result['micro_p']))
logger.info('Micro recall: {}'.format(result['micro_r']))
logger.info('({}, {}, {}, dpr{:.1f})Micro F1: {}'.format(args.dataset, args.bert_name, args.encoder_type, args.dropout_rate, result['micro_f1']))
logger.info('Category-P/R/F1: {}'.format(result['category-p/r/f1']))
