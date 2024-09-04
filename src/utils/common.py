import time

import numpy as np
import torch


def set_seed(seed_val: int  = 42):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def get_unoque_filename(file, ext):
    return time.strftime(f"{file}_%Y_%m_%d_%H_%M.{ext}")



