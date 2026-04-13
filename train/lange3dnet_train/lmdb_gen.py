from __future__ import annotations
import os
import time
import json
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr

from model import LangGeoNetV2
from dataset import create_h5_episode_pathlengths_dataloader
from losses import LangGeoNetLoss

from dataset import convert_h5_to_lmdb


if __name__ == "__main__":
    h5_path= "/media/opervu-user/Data2/ws/data_langgeonet_e3d_action/train_ep500_stratified.h5"
    lmdb_path = "/media/opervu-user/Data2/ws/data_langgeonet_e3d_action/lmdbs/train"

    convert_h5_to_lmdb(h5_path, lmdb_path)