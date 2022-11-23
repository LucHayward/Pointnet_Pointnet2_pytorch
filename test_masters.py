import argparse
import importlib
import os

from Visualisation_utils import visualise_batch, visualise_prediction, turbo_colormap_data
from data_utils.MastersDataset import MastersDataset
import provider

import torch
import datetime
from pathlib import Path
import sys
import logging
import shutil

from tqdm import tqdm
import numpy as np
# import pptk

import wandb

classes = ["keep", "discard"]
sys.path.append("models")

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visualise', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--num_votes', type=int, default=3,
                        help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--data_path', default=None,
                        help='The path to the folder containing the data in .npy formats')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # HYPER PARAMETERS

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu



    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)