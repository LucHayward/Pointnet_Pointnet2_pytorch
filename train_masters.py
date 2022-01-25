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
import pptk

import wandb

classes = ["keep", "discard"]
sys.path.append("models")


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser('Model')
    # Not changed for hparam tuning
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg',
                        help='model name [default: pointnet2_sem_seg]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--data_path', default="data/dummy_big",
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

    def setup_logging_dir():
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
        experiment_dir.mkdir(exist_ok=True)
        checkpoints_dir = experiment_dir.joinpath('checkpoints/')
        checkpoints_dir.mkdir(exist_ok=True)
        log_dir = experiment_dir.joinpath('logs/')
        log_dir.mkdir(exist_ok=True)
        return experiment_dir, log_dir, checkpoints_dir

    def setup_logger():
        # Setup logger (might ditch this)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        log_string('PARAMETER ...')
        log_string(args)

    def setup_data_loaders():

        def _debug_loaders():
            vt = pptk.viewer(all_train_points[:,:3], all_train_labels)
            vv = pptk.viewer(all_val_points[:,:3], all_val_labels)

        log_string("Loading the train dataset")
        TRAIN_DATASET = MastersDataset("train", DATA_PATH, NUM_POINTS, BLOCK_SIZE)
        log_string("Loading the validation dataset")
        VAL_DATASET = MastersDataset("validate", DATA_PATH, NUM_POINTS, BLOCK_SIZE)
        train_data_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
                                                        shuffle=args.shuffle_training_data, num_workers=0,
                                                        pin_memory=True,
                                                        drop_last=True)
        val_data_loader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=BATCH_SIZE,
                                                      shuffle=False, num_workers=0, pin_memory=True,
                                                      drop_last=(True if len(VAL_DATASET) // BATCH_SIZE else False))
        weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

        all_train_points = np.vstack(TRAIN_DATASET.segment_points)
        all_train_labels = np.hstack(TRAIN_DATASET.segment_labels)
        sum_all_train_labels = np.sum(all_train_labels)
        all_val_points = np.vstack(VAL_DATASET.segment_points)
        all_val_labels = np.hstack(VAL_DATASET.segment_labels)
        sum_all_val_labels = np.sum(all_val_labels)
        log_string(f"The size of the training data is {len(TRAIN_DATASET.segment_points)} segments making up "
                   f"{len(TRAIN_DATASET)} samples and a {np.round(sum_all_train_labels/len(all_train_labels), 3)}:{np.round((len(all_train_labels) - sum_all_train_labels)/len(all_train_labels), 3)} label distribution.")
        log_string(f"The size of the validation data is {len(VAL_DATASET.segment_points)} segments making up "
                   f"{len(VAL_DATASET)} samples and a {np.round(sum_all_val_labels/len(all_val_labels),2)}:{np.round((len(all_val_labels) - sum_all_val_labels)/len(all_val_labels),2)} label distribution.")
        # _debug_loaders()
        wandb.config.update({'num_training_data': len(TRAIN_DATASET),
                             'num_test_data': len(VAL_DATASET)})
        return TRAIN_DATASET, VAL_DATASET, train_data_loader, val_data_loader, weights

    def setup_model():
        # Loading the model
        # TODO log this file in wandb
        MODEL = importlib.import_module(args.model)
        shutil.copy('models/%s.py' % args.model, str(experiment_dir))
        shutil.copy('models/pointnet2_utils.py', str(experiment_dir))
        classifier = MODEL.get_model(2, points_vector_size=4).cuda()
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

        return classifier, criterion, optimizer, start_epoch

    def update_lr_momentum():
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        wandb.log({'lr': lr, 'epoch': epoch}, commit=False)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        wandb.log({'bn_momentum': momentum, 'epoch': epoch}, commit=False)
        return lr, momentum

    def post_training_logging_and_vis(all_train_points, all_train_pred, all_train_target):
        if args.log_merged_training_set:
            all_train_points = np.vstack(np.vstack(all_train_points))
            all_train_pred = np.hstack(np.vstack(all_train_pred))
            all_train_target = np.hstack(np.vstack(all_train_target))
            _, unique_indices, unique_counts = np.unique(all_train_points[:, :3], axis=0, return_index=True,
                                                         return_counts=True)
            num_unique_points = len(unique_indices)
            total_training_points = np.vstack(TRAIN_DATASET.segment_points).shape[0]
            print(f"Unique points: {num_unique_points}/{total_training_points} "
                  f"({num_unique_points * 100 // total_training_points}%)")
            visualise_prediction(all_train_points[:, :3], all_train_pred, all_train_target, epoch,
                                 "Train", wandb_section="Visualise-Merged")
        mean_loss = loss_sum / num_batches
        accuracy = total_correct / float(total_seen)
        log_string('Training mean loss: %f' % mean_loss)
        log_string('Training accuracy: %f' % accuracy)
        wandb.log({'Train/mean_loss': mean_loss,
                   'Train/accuracy': accuracy, 'epoch': epoch}, commit=False)
        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'  # Should use .pt
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
            wandb.save(savepath)

    def post_validation_logging_and_vis(all_eval_points, all_eval_pred, all_eval_target, labelweights, best_iou):
        if args.log_merged_validation:
            all_eval_points = np.vstack(np.vstack(all_eval_points))
            all_eval_pred = np.hstack(np.vstack(all_eval_pred))
            all_eval_target = np.hstack(np.vstack(all_eval_target))
            _, unique_indices = np.unique(all_eval_points[:, :3], axis=0, return_index=True)
            num_unique_points = len(unique_indices)
            total_eval_points = np.vstack(VAL_DATASET.room_points).shape[0]
            print(
                f"Unique points: {num_unique_points}/{total_eval_points} ({num_unique_points * 100 // total_eval_points}%)")

            visualise_prediction(all_eval_points[:, :3], all_eval_pred,
                                 all_eval_target, epoch,
                                 "Validation", wandb_section="Visualise-Merged")

        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(
            np.array(total_correct_class) / (np.array(total_iou_denominator_class,
                                                      dtype=np.float) + 1e-6))  # correct prediction/class occurrences + false prediction
        eval_mean_loss = loss_sum / float(num_batches)
        eval_point_accuracy = total_correct / float(total_seen)
        eval_point_avg_class_accuracy = np.mean(
            np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))
        log_string('eval mean loss: %f' % eval_mean_loss)
        log_string('eval point avg class IoU: %f' % mIoU)
        log_string('eval point accuracy: %f' % eval_point_accuracy)
        log_string('eval point avg class acc: %f' % eval_point_avg_class_accuracy)

        wandb.log({'Validation/eval_mean_loss': eval_mean_loss,
                   'Validation/eval_point_mIoU': mIoU,
                   'Validation/eval_point_accuracy': eval_point_accuracy,
                   'Validation/eval_point_avg_class_accuracy': eval_point_avg_class_accuracy,
                   'epoch': epoch}, commit=False)
        # TODO: Want to log:

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                str(l) + ' ' * (14 - 1), labelweights[l - 1],
                total_correct_class[l] / float(total_iou_denominator_class[l]))  # refactor

        # """
        # ------- IoU --------
        # class keep           weight: 0.253, IoU: 0.493
        # class discard        weight: 0.747, IoU: 0.146
        # """
        # wandb.log({"Validation/IoU/": {
        #     "class": seg_label_to_cat[l],
        #     "weight": labelweights[l - 1],
        #     "IoU": total_correct_class[l] / float(total_iou_denominator_class[l]),
        #     "text": iou_per_class_str
        # }
        # }, commit=False)
        # # TODO Custom graph here

        log_string(iou_per_class_str)
        log_string('Eval mean loss: %f' % eval_mean_loss)
        log_string('Eval accuracy: %f' % eval_point_accuracy)

        if mIoU >= best_iou:
            best_iou = mIoU
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'class_avg_iou': mIoU,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
            wandb.save(savepath)
        log_string('Best mIoU: %f' % best_iou)
        return best_iou

    # HYPER PARAMETER
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir, log_dir, checkpoints_dir = setup_logging_dir()
    logger = logging.getLogger("Model")
    setup_logger()

    # Define constants
    DATA_PATH = Path(args.data_path)
    NUM_CLASSES = 2

    NUM_POINTS = args.npoint
    BATCH_SIZE = args.batch_size
    BLOCK_SIZE = args.block_size

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    # Setup data_loaders and classifier
    TRAIN_DATASET, VAL_DATASET, train_data_loader, val_data_loader, weights = setup_data_loaders()
    classifier, criterion, optimizer, start_epoch = setup_model()

    # Training loop
    global_epoch = 0
    best_iou = 0
    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr, momentum = update_lr_momentum()  # TODO how to get around passing in the variables?

        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(train_data_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()  # Set model to training mode

        if not args.validate_only:
            all_train_points, all_train_pred, all_train_target = [], [], []
            for i, (points, target_labels) in tqdm(enumerate(train_data_loader), total=len(train_data_loader),
                                                   desc="Training"):
                optimizer.zero_grad()

                points = points.data.numpy()
                if args.augment_points: points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                points = torch.Tensor(points)
                points, target_labels = points.float().cuda(), target_labels.long().cuda()
                points = points.transpose(2, 1)  # Convert points to num_batches * channels * num_points

                pred_labels, trans_feat = classifier(points)
                pred_labels = pred_labels.contiguous().view(-1, 2)

                # CHECK whats happening here?
                batch_labels = target_labels.view(-1, 1)[:, 0].cpu().data.numpy()
                target_labels = target_labels.view(-1, 1)[:, 0]

                loss = criterion(pred_labels, target_labels, trans_feat, weights)
                loss.backward()
                optimizer.step()

                pred_choice = pred_labels.cpu().data.max(1)[1].numpy()
                correct = np.sum(pred_choice == batch_labels)
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINTS)
                loss_sum += loss

                wandb.log({'Train/inner_epoch_loss_sum': loss_sum,
                           'Train/inner_epoch_accuracy_sum': total_correct / total_seen,
                           'Train/inner_epoch_loss': loss,
                           'Train/inner_epoch_accuracy': correct / len(batch_labels),
                           'epoch': epoch,
                           'Train/inner_epoch_step': (i + epoch * len(train_data_loader))})
                if args.log_merged_training_set:
                    all_train_points.append(np.array(points.transpose(1, 2).cpu()))
                    all_train_pred.append(pred_choice.reshape(args.batch_size, -1))
                    all_train_target.append(np.array(target_labels.cpu()).reshape(args.batch_size, -1))
                # Visualise the first batch in every sample
                if args.log_first_batch_cloud and i == 0:
                    print(f"Visualising Epoch {epoch} Mini-Batch {i}")
                    visualise_batch(np.array(points.transpose(1, 2).cpu()), pred_choice.reshape(args.batch_size, -1),
                                    np.array(target_labels.cpu()).reshape(args.batch_size, -1), i, epoch, 'Train',
                                    pred_labels.exp().cpu().data.numpy(), args.log_merged_training_batches)

            post_training_logging_and_vis(all_train_points, all_train_pred, all_train_target)

        # TODO Validation loop
        with torch.no_grad():
            num_batches = len(train_data_loader)
            total_correct, total_seen, loss_sum = 0, 0, 0

            labelweights = np.zeros(2)  # only used for printing metrics
            total_seen_class, total_correct_class, total_iou_denominator_class = [0, 0], [0, 0], [0, 0]

            classifier = classifier.eval()
            all_eval_points, all_eval_pred, all_eval_target = [], [], []

            log_string('---- EPOCH %03d VALIDATION ----' % (global_epoch + 1))
            for i, (points, target_labels) in tqdm(enumerate(val_data_loader), total=len(val_data_loader),
                                                   desc="Validation"):
                # points = points.data.numpy()
                # points = torch.Tensor(points)
                points, target_labels = points.float().cuda(), target_labels.long().cuda()

                points = points.transpose(2, 1)

                pred_labels, trans_feat = classifier(points)
                pred_labels = pred_labels.contiguous().view(-1, 2)

                # CHECK whats happening here?
                batch_labels = target_labels.view(-1, 1)[:, 0].cpu().data.numpy()
                target_labels = target_labels.view(-1, 1)[:, 0]

                loss = criterion(pred_labels, target_labels, trans_feat, weights)

                pred_choice = pred_labels.cpu().data.max(1)[1].numpy()
                correct = np.sum(pred_choice == batch_labels)
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINTS)
                loss_sum += loss
                tmp, _ = np.histogram(batch_labels, range(NUM_CLASSES + 1))
                labelweights += tmp

                # Logging and visualisation and IoU
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_labels == l))  # How many times the label was in the batch
                    # How often the predicted label was correct in the batch
                    total_correct_class[l] += np.sum((pred_choice == l) & (batch_labels == l))
                    # Class occurrences + total predictions (Union prediction of class (right or wrong) and actual class occurrences.)
                    total_iou_denominator_class[l] += np.sum(((pred_choice == l) | (batch_labels == l)))
                if args.log_merged_validation:
                    all_eval_points.append(np.array(points.transpose(1, 2).cpu()))
                    all_eval_pred.append(pred_choice.reshape(points.shape[0], -1))
                    all_eval_target.append(np.array(target_labels.cpu()).reshape(points.shape[0], -1))
                if args.log_first_batch_cloud and i == 0:
                    visualise_batch(np.array(points.transpose(1, 2).cpu()), pred_choice.reshape(points.shape[0], -1),
                                    np.array(target_labels.cpu()).reshape(points.shape[0], -1), i, epoch, "Validation",
                                    pred_labels.exp().cpu().numpy(), merged=args.log_merged_validation)

            best_iou = post_validation_logging_and_vis(all_eval_points, all_eval_pred, all_eval_target, labelweights,
                                                       best_iou)

        global_epoch += 1
        wandb.log({})

    log_string("Finished")


if __name__ == '__main__':
    args = parse_args()
    os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="Masters", config=args, resume=True, name='mac_testing')
    wandb.run.log_code(".")
    main(args)
    wandb.finish()
