# coding:utf-8
import sys
import torch
import json
import os
import re
import datetime
import argparse
import configparser

sys.path.append('../..')
from pasaie.utils import get_logger
from pasaie.utils.sampler import get_relation_sampler
from pasaie import pasare

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased',
                    help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='',
                    help='Checkpoint name')
parser.add_argument('--pooler', default='entity', choices=['cls', 'entity'],
                    help='Sentence representation pooler')
parser.add_argument('--only_test', action='store_true',
                    help='Only run test')
parser.add_argument('--mask_entity', action='store_true',
                    help='Mask entity mentions')
parser.add_argument('--use_sampler', action='store_true',
                    help='Use sampler')
parser.add_argument('--use_attention', action='store_true',
                    help='whether use attention for DSP and Context feature')
parser.add_argument('--embed_entity_type', action='store_true',
                    help='Embed entity-type information in RE training process')
parser.add_argument('--adv', default='', choices=['fgm', 'pgd', 'flb', 'none'],
        help='embedding adversarial perturbation')
parser.add_argument('--loss', default='ce', choices=['ce', 'focal', 'dice', 'lsr'],
        help='loss function')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                    help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none',
                    # choices=['none', 'semeval', 'wiki80', 'tacred', 'policy', 'nyt10', 'test-policy'],
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
parser.add_argument('--warmup_step', default=0, type=int,
                    help='warmup steps for learning rate scheduler')
parser.add_argument('--max_length', default=256, type=int,
                    help='Maximum sentence length')
parser.add_argument('--max_dsp_path_length', default=15, type=int,
                    help='Maximum entity to root dsp path length')
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
    model_name = 'bert_' + args.pooler + '_dsp_' + args.loss
    if len(args.adv) > 0 and args.adv != 'none':
        model_name += '_' + args.adv
    if args.use_attention:
        model_name += '_attention_cat'
    return model_name
model_name = make_model_name()

# logger
os.makedirs(os.path.join(config['path']['re_log'], args.dataset, model_name), exist_ok=True)
logger = get_logger(sys.argv, os.path.join(config['path']['re_log'], args.dataset, model_name, 
                                        f'{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}.log'))

# tensorboard
os.makedirs(config['path']['re_tb'], exist_ok=True)
tb_logdir = os.path.join(config['path']['re_tb'], args.dataset, model_name, 
                    make_hparam_string(args.optimizer, args.lr, args.batch_size, args.weight_decay, args.max_length))

# Some basic settings
os.makedirs(config['path']['re_ckpt'], exist_ok=True)
if len(args.ckpt) == 0:
    args.ckpt = f'{args.dataset}/{model_name}'
    # args.ckpt = os.path.join(args.dataset, model_name)
ckpt = os.path.join(config['path']['re_ckpt'], f'{args.ckpt}0.pth.tar')
ckpt_cnt = 0
while os.path.exists(ckpt):
    ckpt_cnt += 1
    ckpt = re.sub('\d+\.pth\.tar', f'{ckpt_cnt}.pth.tar', ckpt)

if args.dataset != 'none':
    if 'policy' not in args.dataset:
        pasare.download(args.dataset, root_path=config['path']['input'])
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
    if args.dataset == 'wiki80' or args.dataset == 'fewrel':
        args.metric = 'acc'
    else:
        args.metric = 'micro_f1'
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
if args.pooler == 'entity':
    sentence_encoder = pasare.encoder.BERTEntityWithDSPEncoder(
        pretrain_path=args.pretrain_path,
        max_length=args.max_length,
        max_dsp_path_length=args.max_dsp_path_length if not args.dsp_preprocessed else -1,
        tag2id=tag2id,
        use_attention=args.use_attention,
        mask_entity=args.mask_entity,
        blank_padding=True,
        compress_seq=args.compress_seq,
    )
elif args.pooler == 'cls':
    sentence_encoder = pasare.encoder.BERTWithDSPEncoder(
        pretrain_path=args.pretrain_path,
        max_length=args.max_length,
        max_dsp_path_length=args.max_dsp_path_length if not args.dsp_preprocessed else -1,
        use_attention=args.use_attention,
        mask_entity=args.mask_entity,
        blank_padding=True,
        compress_seq=args.compress_seq
    )
else:
    raise NotImplementedError

# Define the model
model = pasare.model.SoftmaxNN(
    sentence_encoder=sentence_encoder, 
    num_class=len(rel2id), 
    rel2id=rel2id, 
    dropout_rate=args.dropout_rate)

if args.use_sampler:
    sampler = get_relation_sampler(args.train_file, rel2id, 'WeightedRandomSampler')
else:
    sampler = None
# Define the whole training framework
framework = pasare.framework.SentenceWithDSPRE(
    train_path=args.train_file if not args.only_test else None,
    val_path=args.val_file if not args.only_test else None,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    logger=logger,
    tb_logdir=tb_logdir,
    compress_seq=args.compress_seq,
    max_dsp_path_length=args.max_dsp_path_length if args.dsp_preprocessed else -1,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    bert_lr=args.bert_lr,
    weight_decay=args.weight_decay,
    warmup_step=args.warmup_step,
    max_grad_norm=args.max_grad_norm,
    dice_alpha=args.dice_alpha,
    adv=args.adv,
    loss=args.loss,
    opt=args.optimizer,
    sampler=sampler
)
if ckpt_cnt > 0:
    framework.load_state_dict(torch.load(re.sub('\d+\.pth\.tar', f'{ckpt_cnt-1}.pth.tar', ckpt)))
# Train the model
if not args.only_test:
    framework.train_model('micro_f1')

# Test
if not args.only_test:
    framework.load_state_dict(torch.load(ckpt))
result = framework.eval_model(framework.test_loader)

# Print the result
logger.info('Test set results:')
logger.info('Accuracy: {}'.format(result['acc']))
logger.info('Micro precision: {}'.format(result['micro_p']))
logger.info('Micro recall: {}'.format(result['micro_r']))
logger.info('Micro F1: {}'.format(result['micro_f1']))
