import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet_unfrozen_l3 import ResNetRegressorUnfrozenl3
import pandas as pd
from models.dataset import CensusDataset
from torch.utils.data import DataLoader
import torch
from models.collate import collate_fn
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from src.splitting import split_by_tract




