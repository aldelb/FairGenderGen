
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl
import sklearn

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('whitegrid')

#disable debbuging API for faster training
torch.autograd.profiler.profile(False)
torch.autograd.set_detect_anomaly(False)
# enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

seed = 2855
random.seed(seed)
np.random.seed(seed)
sklearn.utils.check_random_state(seed)
torch.manual_seed(seed)
pl.seed_everything(seed)

from constants.constants_utils import read_params
from evaluate import Evaluate
from train import Train
from generate import Generate
from visualize_data import VisualizeData
from classify import Classify


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-id', help='Path to save result and models', default="0")
    parser.add_argument('-task', help='choose a task beetween "train", "generate", "evaluate" and "gen_eval"', default="train")

    #for generation and evaluation
    parser.add_argument('-dataset', help='which video we want to generate', default="")
    parser.add_argument('-file', help='which file we want to generate (one by one)', default="")
    parser.add_argument('-epoch', help='number of epoch of recovred model', default=1000)
    parser.add_argument('-all_step', help='number of epoch of recovred model', action='store_true')

    #for evaluation
    parser.add_argument('-dtw', action='store_true')
    parser.add_argument('-pca', action='store_true')
    parser.add_argument('-curve', action='store_true')
    parser.add_argument('-curveVideo', action='store_true')
    parser.add_argument('-motion', action='store_true')

    parser.add_argument('-label', default='')

    args = parser.parse_args()

    task = args.task
    read_params(args.params, task, args.id)

    if(task == "train"):
        train = Train()
    elif(task == "generate"):
        generate = Generate(args.dataset, args.epoch, None)
    elif(task == "generate_one_file"):
        generate = Generate(args.dataset, args.epoch, args.file)

    
