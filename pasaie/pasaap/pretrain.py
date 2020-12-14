import os
import configparser
import torch

from . import encoder
from . import model
from .import framework

project_path = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-3])
config = configparser.ConfigParser()
config.read(os.path.join(project_path, 'config.ini'))
default_root_path = config['path']['ap_ckpt']


def get_model(dataset_name,
              model_name,
              root_path=default_root_path):
    abs_ckpt_path = os.path.join(root_path, dataset_name, model_name + '.pth.tar')
    if dataset_name in ['sentence_importance_judgement']:
        sij_model = torch.load(abs_ckpt_path, map_location='cpu')
        return sij_model
    else:
        raise NotImplementedError
