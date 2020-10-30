import torch
import numpy as np
import random

def fix_seed(seed=12345): 
    """fix random seed for reproduction

    Args:
        seed (int): [description]. Defaults to 12345.
    """
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) # gpu
    np.random.seed(seed) # numpy
    random.seed(seed) # random and transforms
    torch.backends.cudnn.deterministic=True # cudnn