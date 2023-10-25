import sys
import argparse
import time
from datetime import datetime
import math
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch import Tensor
import torch.nn.functional as F

from tqdm import tqdm

from models import model_initialization
from datasets import dataset_initialization

def main():
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser()

    ### Tensors type ###
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")

    ### Dataset ###
    parser.add_argument("--data_size", type=int, default=100000)
    parser.add_argument("--embedding_dim", type=int, default=10)
    parser.add_argument("--one_hot", action="store_true")
    parser.add_argument("--dummy", action="store_true")

    ### Model ###
    parser.add_argument("--attention_type", type=str, default="standard")
    parser.add_argument("--reducer_type", type=str, default="MLP")
    parser.add_argument("--layer_norm", type=bool, default=True)
    parser.add_argument("--residual", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    
    model = model_initialization(args=args)
    vocab, trainset = dataset_initialization(args=args)






