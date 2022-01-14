import argparse
import importlib
import os

from Visualisation_utils import visualise_batch, visualise_prediction, turbo_colormap_data
from data_utils.MastersDataset import MastersDataset

import torch
import datetime
from pathlib import Path
import sys
import logging
import shutil

from tqdm import tqdm
import numpy as np
import time
import pptk

import wandb

classes = ["keep", "discard"]
sys.path.append("models")


def inplace_relu(m):
    classname = m.__clas__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser('Model')
    # Not changed for hparam tuning
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg',
                        help='model name [default: pointnet2_sem_seg]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--data_path', default="data/PatrickData/Church/MastersFormat/dummy",
                        help='The path to the folder containing the data in .npy formats')
    parser.add_argument('--log_dir', type=str, default=None, help="Log path [default: None]")

    # Tune these early and likely not change much
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

    # Common hparams to be tuned
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--npoint', type=int, default=4096,
                        help='Number of points in each column to process at one time [default: 4096]')
    parser.add_argument('--block_size', default=1.0, type=float, help='column size for sampling')
    parser.add_argument('--augment_points', action='store_true', help='Augment pointcloud (currently by rotation)')

    # Logging/Visualisation parameters
    parser.add_argument('--log_first_batch_cloud', action='store_true', help='Log the pointclouds that get sampled')
    parser.add_argument('--log_merged_validation', action='store_true', help='Log the merged validation pointclouds')
    parser.add_argument('--log_merged_training_batches', action='store_true',
                        help='When logging training batch visualisations, merge batches in global coordinate space before visualising.')
    parser.add_argument('--log_merged_training_set', action='store_true',
                        help='merge all teh points used during training into one visualisation')

    # Debugging parameters
    parser.add_argument('--validate_only', action='store_true', help='Skip training and only run the validation step')
    parser.add_argument('--shuffle_training_data', action='store_true', help='Shuffle the training data loader')

    # Exposing model hparams
    # Pointnet Set Abstraction: Group All options
    parser.add_argument('--psa1_group_all', action='store_true',
                        help='Use Group_all in Pointnet Set Abstraction Layer 1')
    parser.add_argument('--psa2_group_all', action='store_true',
                        help='Use Group_all in Pointnet Set Abstraction Layer 2')
    parser.add_argument('--psa3_group_all', action='store_true',
                        help='Use Group_all in Pointnet Set Abstraction Layer 3')
    parser.add_argument('--psa4_group_all', action='store_true',
                        help='Use Group_all in Pointnet Set Abstraction Layer 4')

    # Pointnet Set Abstraction: Sphere Radius
    parser.add_argument('--psa1_radius', default=0.1, help='Sphere lookup radius in Pointnet Set Abstraction Layer 1')
    parser.add_argument('--psa2_radius', default=0.2, help='Sphere lookup radius in Pointnet Set Abstraction Layer 2')
    parser.add_argument('--psa3_radius', default=0.4, help='Sphere lookup radius in Pointnet Set Abstraction Layer 3')
    parser.add_argument('--psa4_radius', default=0.8, help='Sphere lookup radius in Pointnet Set Abstraction Layer 4')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    def weights_init(m):
        # CHECK what is this for
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def setup_logging_dir(args):
        # Create logging directory
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        experiment_dir = Path('./log/')
        experiment_dir.mkdir(exist_ok=True)
        experiment_dir = experiment_dir.joinpath('masters')
        experiment_dir.mkdir(exist_ok=True)
        if args.log_dir is None:
            experiment_dir = experiment_dir.joinpath(timestr)
        else:
            experiment_dir = experiment_dir.joinpath(args.log_dir)
        experiment_dir = experiment_dir.joinpath(timestr)
        experiment_dir.mkdir(exist_ok=True)
        checkpoints_dir = experiment_dir.joinpath('checkpoints/')
        checkpoints_dir.mkdir(exist_ok=True)
        log_dir = experiment_dir.joinpath('logs/')
        log_dir.mkdir(exist_ok=True)
        return experiment_dir, log_dir

    def setup_logger(args, log_dir):
        # Setup logger (might ditch this)
        global logger
        logger = logging.getLogger("Model")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        log_string('PARAMETER ...')
        log_string(args)

    def setup_data_loaders(BATCH_SIZE, BLOCK_SIZE, DATA_PATH, NUM_POINTS, args, log_string):
        log_string("Loading the train dataset")
        TRAIN_DATASET = MastersDataset("train", DATA_PATH, NUM_POINTS, BLOCK_SIZE)
        log_string("Loading the validation dataset")
        VAL_DATASET = MastersDataset("validation", DATA_PATH, NUM_POINTS, BLOCK_SIZE)
        train_data_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
                                                        shuffle=args.shuffle_training_data, num_workers=0,
                                                        pin_memory=True,
                                                        drop_last=True)
        val_data_loader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=BATCH_SIZE,
                                                      shuffle=False, num_workers=0, pin_memory=True,
                                                      drop_last=(True if len(VAL_DATASET) // BATCH_SIZE else False))
        weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
        log_string(f"The size of the training data is {len(TRAIN_DATASET)} segments")
        log_string(f"The size of the validation data is {len(VAL_DATASET)} segments")
        wandb.config.update({'num_training_data': len(TRAIN_DATASET),
                             'num_test-data': len(VAL_DATASET)})
        return TRAIN_DATASET, VAL_DATASET, train_data_loader, val_data_loader, weights

    def setup_model(args, experiment_dir, log_string, weights_init):
        # Loading the model
        # TODO log this file in wandb
        MODEL = importlib.import_module(args.model)
        shutil.copy('models/%s.py' % args.model, str(experiment_dir))
        shutil.copy('models/pointnet2_utils.py', str(experiment_dir))
        classifier = MODEL.get_model(2).cuda()
        criterion = MODEL.get_loss().cuda()
        classifier.apply(inplace_relu)
        wandb.watch(classifier, criterion, log='all', log_freq=10)
        # Check for models that have already been trained.
        try:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model')
        except:
            log_string('No existing model, starting training from scratch...')
            start_epoch = 0
            classifier = classifier.apply(weights_init)
        # Setup otpimizer
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.decay_rate
            )
        else:
            optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

        return classifier, criterion, optimizer

    # HYPER PARAMETER
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir, log_dir = setup_logging_dir(args)
    setup_logger(args, log_dir, log_string)

    # Define constants
    DATA_PATH = args.data_path
    NUM_CLASSES = 2

    NUM_POINTS = args.npoint
    BATCH_SIZE = args.batch_size
    BLOCK_SIZE = args.block_size

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    # Setup data_loaders and classifier
    TRAIN_DATASET, VAL_DATASET, train_data_loader, val_data_loader, weights = setup_data_loaders(BATCH_SIZE, BLOCK_SIZE,
                                                                                                 DATA_PATH, NUM_POINTS,
                                                                                                 args, log_string)
    classifier, criterion, optimizer = setup_model(args, experiment_dir, log_string, weights_init)


    # Training loop
    global_epoch = 0
    best_iou = 0






if __name__ == '__main__':
    args = parse_args()
    wandb.init(project="Masters", config=args, resume=True, name='Church-baseline')
