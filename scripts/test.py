import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet_unfrozen_l3 import ResNetRegressorUnfrozenl3
import pandas as pd
from models.dataset_multi import CensusDataset
from torch.utils.data import DataLoader
import torch
from models.collate import collate_fn
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from src.splitting import split_by_tract


# This file is a scratchpad — imports are set up but no code runs here yet.
# You can drop quick tests in here without creating a new file.
# Just uncomment whatever you need and run it.