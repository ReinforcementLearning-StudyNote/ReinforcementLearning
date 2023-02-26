# import random
# import torch.nn as nn
# import torch.nn.functional as func
# import torch
from environment.config.xml_write import xml_cfg
from common.common_func import *
from common.common_cls import *
import pandas as pd

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class A3C:
    def __init__(self):
        pass
