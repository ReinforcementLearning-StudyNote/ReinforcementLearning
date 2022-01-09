import os
import time
import numpy as np
import math
from environment.config.xml_write import xml_cfg
import random
from common.common import *
import torch

# a = torch.tensor([1,2,3,4,4]).detach().numpy()
# print(np.argmax(a))
rad_bet_pos_vel = np.arccos(2)
print(rad_bet_pos_vel)