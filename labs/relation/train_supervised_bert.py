# coding:utf-8
import sys
import torch
import json
import os
import argparse
import configparser
import datetime

sys.path.append('../..')
from pasaie.utils import get_logger
import pasaie
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
parser.add_argument('--embed_entity_type', default=True,
                    help='Embed entity-type information in RE training process')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                    help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none',
                    choices=['none', 'semeval', 'wiki80', 'tacred', 'policy', 'nyt10', 'test-policy'],
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

# Hyper-parameters
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
                    help='Learning rate')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='optimizer')
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

# logger
os.makedirs(config['path']['re_log'], exist_ok=True)
os.makedirs(os.path.join(config['path']['re_log'], f'{args.dataset}_bert_{args.pooler}'), exist_ok=True)
logger = get_logger(sys.argv, os.path.join(config['path']['re_log'], f'{args.dataset}_bert_{args.pooler}',
                                           f'{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}.log'))


# tensorboard
def make_hparam_string(op, lr, bs, wd, ml):
    return "%s_lr_%.0E,bs=%d,wd=%.0E,ml=%d" % (op, lr, bs, wd, ml)


os.makedirs(config['path']['re_tb'], exist_ok=True)
tb_logdir = os.path.join(config['path']['re_tb'],
                         f'{args.dataset}_bert_{args.pooler}/{make_hparam_string(args.optimizer, args.lr, args.batch_size, args.weight_decay, args.max_length)}')

# Some basic settings
os.makedirs(config['path']['re_ckpt'], exist_ok=True)
if len(args.ckpt) == 0:
    args.ckpt = '{}_{}_{}'.format(args.dataset, 'bert', args.pooler)
ckpt = os.path.join(config['path']['re_ckpt'], f'{args.ckpt}.pth.tar')
ckpt_cnt = 1
while os.path.exists(ckpt):
    ckpt = ckpt.replace('.pth.tar', f'{ckpt_cnt}.pth.tar')
    ckpt_cnt += 1

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
        logger.warn("Test file {} does not exist! Use val file instead".format(args.test_file))
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

tag2id = json.load(open(args.tag2id_file))
rel2id = None if not args.embed_entity_type else json.load(open(args.rel2id_file))

# Define the sentence encoder
if args.pooler == 'entity':
    sentence_encoder = pasaie.pasare.encoder.BERTEntityEncoder(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        tag2id=tag2id,
        mask_entity=args.mask_entity,
        blank_padding=True
    )
elif args.pooler == 'cls':
    sentence_encoder = pasaie.pasare.encoder.BERTEncoder(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity,
        blank_padding=True
    )
else:
    raise NotImplementedError

# Define the model
model = pasaie.pasare.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
if args.use_sampler:
    sampler = pasaie.pasare.framework.get_sampler(args.train_file, rel2id, 'WeightedRandomSampler')
else:
    sampler = None
# Define the whole training framework
framework = pasaie.pasare.framework.SentenceRE(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    logger=logger,
    tb_logdir=tb_logdir,
    compress_seq=args.compress_seq,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    opt=args.optimizer,
    warmup_step=args.warmup_step,
    sampler=sampler
)

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
