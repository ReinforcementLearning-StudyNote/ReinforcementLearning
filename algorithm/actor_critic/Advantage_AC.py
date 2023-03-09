import pandas as pd
import torch

from common.common_func import *
from common.common_cls import *
from environment.config.xml_write import xml_cfg

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class Advantage_AC:
	def __init__(self,
				 gamma: float = 0.9,
                 noise_clip: float = 1 / 2,
                 noise_policy: float = 1 / 4,
                 policy_delay: int = 5,
                 critic1_soft_update: float = 1e-2,
                 critic2_soft_update: float = 1e-2,
                 actor_soft_update: float = 1e-2,
                 memory_capacity: int = 5000,
                 batch_size: int = 64,
                 modelFileXML: str = '',
                 path: str = ''):
		pass
