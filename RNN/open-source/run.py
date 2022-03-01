import time 
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from tensorboardx import SummaryWriter

parser = argparse.ArgumentParser(description='chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help = 'choose a model')
parser.add_argument('--embedding', default='pre_trained', type=str, help='')
parser.add_argument('--word', default=False, type=bool, help = '')
args = parser.parse_args()

if __name__ == '__main__':
    datasets = 'THUCNews'

    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding == 'random'
    model_name = args.model

    if args.name == 'FastText':
        from utils_fasttext import build