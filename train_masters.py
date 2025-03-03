import argparse
import datetime
import importlib
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm import tqdm

import provider
from Visualisation_utils import visualise_batch, create_confusion_mask
from data_utils.MastersDataset import MastersDataset

# import pptk

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

    # Active Learning
    parser.add_argument('--active_learning', default=False)
    parser.add_argument('--save_best_train_model', default=False,
                        help='Save the best model from the training based on mIoU')
    parser.add_argument('--validation_repeats', default=1,
                        help='How many times to repeat the validation classification')

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
    parser.add_argument('--force_even', action='store_true',
                        help='Force the label distribution per batch to be approximately even')
    parser.add_argument('--sample_all_validation', action='store_true',
                        help='Samples all the points in a grid for the validation')
    parser.add_argument('--relative_point_coords', action='store_true',
                        help='Use the old S3DIS point features (relativeXYZ,IGB,XYZ/max(xyz)')
    parser.add_argument('--save_intermediate_models', default=True,
                        help='Save models at each epoch instead of only the "best"')

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
    parser.add_argument('--sa1_radius', default=0.1, help='Sphere lookup radius in Pointnet Set Abstraction Layer 1')
    parser.add_argument('--sa2_radius', default=0.2, help='Sphere lookup radius in Pointnet Set Abstraction Layer 2')
    parser.add_argument('--sa3_radius', default=0.4, help='Sphere lookup radius in Pointnet Set Abstraction Layer 3')
    parser.add_argument('--sa4_radius', default=0.8, help='Sphere lookup radius in Pointnet Set Abstraction Layer 4')

    return parser.parse_args()


def setup_logging_dir(config, exp_dir='masters'):
    # Create logging directory
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(exp_dir)
    experiment_dir.mkdir(exist_ok=True)
    if config["log_dir"] is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        if "log/active_learning" in str(config["log_dir"]):
            experiment_dir = Path(config["log_dir"])
        else:
            experiment_dir = experiment_dir.joinpath(config["log_dir"])
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    return experiment_dir, log_dir, checkpoints_dir


def setup_logger(logger, log_dir, config):
    # Setup logger (might ditch this)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, config["model"]))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def setup_wandb_classification_metrics():
    # Train classification metrics
    wandb.define_metric('Train/TP', summary='max')
    wandb.define_metric('Train/FP', summary='min')
    wandb.define_metric('Train/TN', summary='max')
    wandb.define_metric('Train/FN', summary='min')

    wandb.define_metric('Train/category-TP', summary='max')
    wandb.define_metric('Train/category-FP', summary='min')
    wandb.define_metric('Train/category-TN', summary='max')
    wandb.define_metric('Train/category-FN', summary='min')

    wandb.define_metric('Train/Precision', summary='max')
    wandb.define_metric('Train/Recall', summary='max')
    wandb.define_metric('Train/F1', summary='max')
    wandb.define_metric('Train/mIoU', summary='max')
    wandb.define_metric('Train/accuracy', summary='max')
    wandb.define_metric('Train/mean_loss', summary='min')

    # Validation Classification metrics
    wandb.define_metric('validation/TP', summary='max')
    wandb.define_metric('validation/FP', summary='min')
    wandb.define_metric('validation/TN', summary='max')
    wandb.define_metric('validation/FN', summary='min')

    wandb.define_metric('validation/category-TP', summary='max')
    wandb.define_metric('validation/category-FP', summary='min')
    wandb.define_metric('validation/category-TN', summary='max')
    wandb.define_metric('validation/category-FN', summary='min')

    wandb.define_metric('Validation/Precision', summary='max')
    wandb.define_metric('Validation/Recall', summary='max')
    wandb.define_metric('Validation/F1', summary='max')
    wandb.define_metric('Validation/eval_point_mIoU', summary='max')
    wandb.define_metric('Validation/eval_point_accuracy', summary='max')
    wandb.define_metric('Validation/eval_point_avg_class_accuracy', summary='max')
    wandb.define_metric('Validation/eval_mean_loss', summary='min')


def _log_string(str, logger):
    logger.info(str)
    print(str)

def main(config):
    def setup_wandb_metrics():
        # Inner epoch steps
        wandb.define_metric('Train/inner_epoch_step')
        wandb.define_metric('Train/inner_epoch/*', step_metric="Train/inner_epoch_step")

        wandb.define_metric('Validation/inner_epoch_step')
        wandb.define_metric('Validation/inner_epoch/*', step_metric="Validation/inner_epoch_step")

        # Debugging metrics
        wandb.define_metric('Validation/top10variance_avg', summary='mean')

        setup_wandb_classification_metrics()

    def log_string(str):
        _log_string(str, logger)

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

    def setup_data_loaders():

        # def _debug_loaders():
            # vt = pptk.viewer(all_train_points[:, :3], all_train_labels)
            # vv = pptk.viewer(all_val_points[:, :3], all_val_labels)

        log_string("Loading the train dataset")
        TRAIN_DATASET = MastersDataset("train", DATA_PATH, NUM_POINTS, BLOCK_SIZE, force_even=config["force_even"],
                                       sample_all_points=True, relative_coords=config["relative_point_coords"])
        log_string("Loading the validation dataset")
        VAL_DATASET = MastersDataset("validate", DATA_PATH, NUM_POINTS, BLOCK_SIZE, force_even=config["force_even"],
                                     sample_all_points=True, relative_coords=config["relative_point_coords"])
        train_data_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
                                                        shuffle=config["shuffle_training_data"], num_workers=0,
                                                        pin_memory=True,
                                                        drop_last=True)

        val_data_loader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=BATCH_SIZE,
                                                      shuffle=False, num_workers=0, pin_memory=True,
                                                      drop_last=False)
        if config["force_even"]:
            TRAIN_DATASET.batch_label_counts = np.zeros(BATCH_SIZE)
            VAL_DATASET.batch_label_counts = np.zeros(BATCH_SIZE)

        weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

        all_train_points = np.vstack(TRAIN_DATASET.segment_points)
        all_train_labels = np.hstack(TRAIN_DATASET.segment_labels)
        sum_all_train_labels = np.sum(all_train_labels)
        all_val_points = np.vstack(VAL_DATASET.segment_points)
        all_val_labels = np.hstack(VAL_DATASET.segment_labels)
        sum_all_val_labels = np.sum(all_val_labels)
        log_string(f"The size of the training data is {len(TRAIN_DATASET.segment_points)} segments making up "
                   f"{len(TRAIN_DATASET)} samples and a {np.round(sum_all_train_labels / len(all_train_labels), 3)}:{np.round((len(all_train_labels) - sum_all_train_labels) / len(all_train_labels), 3)} label distribution.")
        log_string(f"The size of the validation data is {len(VAL_DATASET.segment_points)} segments making up "
                   f"{len(VAL_DATASET)} samples and a {np.round(sum_all_val_labels / len(all_val_labels), 2)}:{np.round((len(all_val_labels) - sum_all_val_labels) / len(all_val_labels), 2)} label distribution.")
        # _debug_loaders()
        wandb.config.update({'num_training_data': len(TRAIN_DATASET),
                             'num_test_data': len(VAL_DATASET)}, allow_val_change=True)
        return TRAIN_DATASET, VAL_DATASET, train_data_loader, val_data_loader, weights

    def setup_model():
        # Loading the model
        # TODO log this file in wandb
        MODEL = importlib.import_module(config["model"])
        shutil.copy('models/%s.py' % config["model"], str(experiment_dir))
        shutil.copy('models/pointnet2_utils.py', str(experiment_dir))
        # classifier = MODEL.get_model(2, points_vector_size=(9 if config["relative_point_coords"] else 4)).cuda()
        classifier = MODEL.get_model(2, points_vector_size=(3)).cuda()
        criterion = MODEL.get_loss().cuda()
        classifier.apply(inplace_relu)
        wandb.watch(classifier, criterion, log='all', log_freq=10)
        checkpoint_path = str(experiment_dir) + '/checkpoints/best_model.pth'
        loaded_pretrained = True
        # Check for models that have already been trained.
        try:
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model')
            # if config["only_train_last_two_layers"]:
            #     for param in classifier.parameters():
            #         param.requires_grad = False
            #     classifier.conv1.weight.requires_grad = True
            #     classifier.conv2.weight.requires_grad = True
            #     classifier.bn1.weight.requires_grad = True
            #     classifier.conv1.bias.requires_grad = True
            #     classifier.conv2.bias.requires_grad = True
            #     classifier.bn1.bias.requires_grad = True


        except:
            log_string(f'No existing model, starting training from scratch...({checkpoint_path})')
            loaded_pretrained = False
            start_epoch = 0
            classifier = classifier.apply(weights_init)

        # Setup otpimizer
        if config["optimizer"] == 'Adam':
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=config["learning_rate"],
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=config["decay_rate"]
            )
            if loaded_pretrained:
                log_string("Loading optimizer state dict")
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except KeyError:
                    log_string("No optimizer state dict found, initializing as normal")
        else:
            optimizer = torch.optim.SGD(classifier.parameters(), lr=config["learning_rate"], momentum=0.9)

        return classifier, criterion, optimizer, start_epoch

    def update_lr_momentum():
        lr = max(config["learning_rate"] * (config["lr_decay"] ** (epoch // config["step_size"])), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        wandb.log({'lr': lr}, commit=False)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        wandb.log({'bn_momentum': momentum}, commit=False)
        return lr, momentum

    def post_training_logging_and_vis(all_train_points, all_train_pred, all_train_target, best_train_iou):
        unique_indices = None
        if config["log_merged_training_set"]:
            all_train_points = np.vstack(np.vstack(all_train_points))
            all_train_pred = np.hstack(np.vstack(all_train_pred))
            all_train_target = np.hstack(np.vstack(all_train_target))
            unique_points, unique_indices, unique_counts = np.unique(all_train_points[:, :3], axis=0, return_index=True,
                                                                     return_counts=True)
            unique_indices.sort()
            unique_points = all_train_points[unique_indices, :3]
            num_unique_points = len(unique_indices)

            train_dataset_points = np.vstack(TRAIN_DATASET.segment_points)
            train_dataset_points = train_dataset_points.astype('float32')
            # trained_idxs = (np.isin(train_dataset_points[:, 0], unique_points[:, 0]) & np.isin(train_dataset_points[:, 1], unique_points[:, 1]) & np.isin(
            #     train_dataset_points[:, 2], unique_points[:, 2])).nonzero()
            # trained_idxs = trained_idxs[0]
            # trained_mask = np.isin(train_dataset_points[:, 0], unique_points[:, 0]) & np.isin(train_dataset_points[:, 1], unique_points[:, 1]) & np.isin(
            #     train_dataset_points[:, 2], unique_points[:, 2])
            # v = pptk.viewer(train_dataset_points[:, :3], np.hstack(TRAIN_DATASET.segment_labels), trained_mask)
            # vmissed = pptk.viewer(train_dataset_points[~trained_mask, :3], np.hstack(TRAIN_DATASET.segment_labels)[~trained_mask])

            total_training_points = np.vstack(TRAIN_DATASET.segment_points).shape[0]
            print(f"Unique points sampled: {num_unique_points}/{total_training_points} "
                  f"({num_unique_points * 100 // total_training_points}%)")

            # visualise_prediction(all_train_points[:, :3], all_train_pred, all_train_target, epoch,
            #                      "Train", wandb_section="Visualise-Merged")
            tn, tp, fn, fp = \
                np.histogram(create_confusion_mask(all_train_points, all_train_pred, all_train_target),
                             [0, 1, 2, 3, 4])[0]
            precision = tp / (tp + fp)
            recall = tp / (tp + fp)
            f1 = 2 * (recall * precision) / (recall + precision)
            percentage_category_confusion = [round(tp / (tp + fn), 3), round(fp / (tn + fp), 3),
                                             round(fn / (tp + fn), 3), round(tn / (tn + fp), 3)]
            # Could do this with sklearn.metrics.confusion_matrix(normalise='true')

            wandb.log({'Train/confusion_matrix': wandb.plot.confusion_matrix(probs=None, y_true=all_train_target,
                                                                             preds=all_train_pred,
                                                                             class_names=["keep", "discard"]),
                       'Train/TP': tp, 'Train/FP': fp,
                       'Train/FN': fn, 'Train/TN': tn,
                       'Train/category-TP': percentage_category_confusion[0],
                       'Train/category-FP': percentage_category_confusion[1],
                       'Train/category-FN': percentage_category_confusion[2],
                       'Train/category-TN': percentage_category_confusion[3],
                       'Train/Precision': precision, 'Train/Recall': recall, 'Train/F1': f1, }, commit=False)

        mean_loss = loss_sum / num_batches
        accuracy = total_correct / float(total_seen)

        mIoU = np.mean(
            np.array(total_correct_class) / (np.array(total_iou_denominator_class,
                                                      dtype=np.float) + 1e-6))  # correct prediction/class occurrences + false prediction

        log_string('Training mean loss: %f' % mean_loss)
        log_string('Training accuracy: %f' % accuracy)
        log_string('Training mIoU: %f' % mIoU)
        wandb.log({'Train/mean_loss': mean_loss,
                   'Train/accuracy': accuracy,
                   'Train/mIoU': mIoU}, commit=False)

        if config["active_learning"] or config["save_intermediate_models"]:
            log_string("Saving intermediary training model")
            savepath = str(checkpoints_dir) + f'model_epoch{epoch}.pth'
            state = {
                'epoch': epoch,
                'class_avg_iou': mIoU,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string("Saved intermediary training model")

        if mIoU > best_train_iou:
            best_train_iou = mIoU
            if config["save_best_train_model"]:
                nonlocal SAVE_CURRENT_EPOCH_PREDS
                SAVE_CURRENT_EPOCH_PREDS = True

                if config["active_learning"]:
                    # Save the best model training predictions thus far incase we want them later for AL visualisation
                    log_string('Save model training predictions...')
                    savepath = str(experiment_dir) + '/train_predictions.npz'
                    log_string('Saving at %s' % savepath)
                    np.savez_compressed(savepath, points=all_train_points[unique_indices], preds=all_train_pred[
                        unique_indices], target=all_train_target[unique_indices])
                    shutil.copy(savepath, savepath[:-4] + f'_epoch{epoch}.npz')
                    log_string('Saved model training predictions.')

                log_string('Save best train model...')
                savepath = str(checkpoints_dir) + '/best_train_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saved best train model.')
                wandb.save(savepath)

        if epoch % 5 == 0:
            log_string('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'  # Should use .pt
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'class_avg_iou': mIoU,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saved model....')
            wandb.save(savepath)

        return best_train_iou

    def post_validation_logging_and_vis(all_eval_points, all_eval_pred, all_eval_target, labelweights, best_iou,
                                        all_eval_variance, all_eval_features):
        if config["log_merged_validation"]:
            unique_points, unique_indices = np.unique(all_eval_points[:, :3], axis=0, return_index=True)
            unique_indices.sort()
            unique_points = all_eval_points[unique_indices, :3]
            # unique_preds = np.copy(all_eval_pred[unique_indices])
            num_unique_points = len(unique_indices)
            total_eval_points = np.vstack(VAL_DATASET.segment_points).shape[0]
            print(f"Unique points: {num_unique_points}/{total_eval_points} "
                  f"({num_unique_points * 100 // total_eval_points}%)")

            # validation_dataset_points = np.vstack(VAL_DATASET.segment_
            #    points)

            # Save the model validation predictions (from the best training model so far) for AL later
            if SAVE_CURRENT_EPOCH_PREDS:
                log_string('Save model validation predictions...')
                savepath = str(experiment_dir) + '/val_predictions.npz'
                log_string('Saving at %s' % savepath)

                # Need to combine the variances down to len(VAL_DATASET.grid_cell_to_segment)
                # number of cell-batches to each cell in the grid.
                samples_per_cell = np.array(VAL_DATASET.grid_cell_to_segment) // NUM_POINTS
                # Collect the variances together based on the GRID_CELLs they represent
                cell_variance, cell_features = [], []
                samples_per_cell_enumerator = enumerate(samples_per_cell)
                for idx, num_samples in samples_per_cell_enumerator:
                    if num_samples == 1:
                        cell_variance.append(all_eval_variance[idx])
                        cell_features.append(all_eval_features[idx])
                    else:
                        cell_variance.append(np.mean(all_eval_variance[idx:idx + num_samples]))
                        cell_features.append(np.mean(all_eval_features[idx:idx + num_samples], axis=0))

                cell_variance = np.array(cell_variance)
                cell_features = np.array(cell_features)
                cell_variance = cell_variance / cell_variance.sum()  # Normalise to [-1,1]
                cell_features = cell_features / cell_features.sum()  # Normalise to [-1,1]

                np.savez_compressed(savepath, points=all_eval_points[unique_indices],
                                    preds=all_eval_pred[unique_indices],
                                    target=all_eval_target[unique_indices], variance=cell_variance,
                                    point_variance=np.repeat(cell_variance, VAL_DATASET.grid_cell_to_segment)[
                                        unique_indices],
                                    grid_mask=VAL_DATASET.grid_mask, features=cell_features,
                                    # samples_per_cell=samples_per_cell
                                    )
                import shutil
                shutil.copy(savepath, savepath[:-4] + f'_epoch{epoch}.npz')
                log_string('Saved model validation predictions.')
                np.sort(cell_variance)
                wandb.log({'Validation/top10variance_avg': np.mean(cell_variance[:10])}, commit=False)

            # validation_dataset_points = validation_dataset_points.astype('float32')
            # trained_idxs = (np.isin(validation_dataset_points[:, 0], unique_points[:, 0]) & np.isin(validation_dataset_points[:, 1], unique_points[:, 1]) & np.isin(
            #     validation_dataset_points[:, 2], unique_points[:, 2])).nonzero()
            # trained_idxs = trained_idxs[0]
            # trained_mask = np.isin(validation_dataset_points[:, 0], unique_points[:, 0]) & np.isin(validation_dataset_points[:, 1], unique_points[:, 1]) & np.isin(
            #     validation_dataset_points[:, 2], unique_points[:, 2])
            # v = pptk.viewer(validation_dataset_points[:, :3], np.hstack(VAL_DATASET.segment_labels), trained_mask)
            # vmissed = pptk.viewer(validation_dataset_points[~trained_mask, :3], np.hstack(VAL_DATASET.segment_labels)[~trained_mask])

            # from collections import Counter
            # preds = {}
            # cnt = 0
            # multiclassified_idxs = []
            # voted_preds = np.copy(all_eval_pred)
            # for i in tqdm(range(len(all_eval_pred))):
            #     preds.setdefault(tuple(all_eval_points[i, :3]), []).append(
            #         np.array((all_eval_pred[i], all_eval_target[i], i)))
            # for item in tqdm(preds.items()):
            #     first = np.array(item[1])
            #     cnt += len(np.unique(first[:,:2])) - 1
            #     if len(np.unique(first[:,:2])) > 1: multiclassified_idxs += first[:,2].tolist()
            #     voted_preds[first[:,2]] = (Counter(first[:,0]).most_common(1)[0][0])
            # print(f"Points with different results: {cnt} ({cnt * 100 / num_unique_points:.2f}%)")

            # visualise_prediction(all_eval_points[:, :3], all_eval_pred,
            #                      all_eval_target, epoch,
            #                      "Validation", wandb_section="Visualise-Merged")

            # CHECK why doesn''t the below work nicely
            # confusion_matrix = confusion_matrix(all_eval_target, all_eval_pred)
            tn, tp, fn, fp = \
                np.histogram(create_confusion_mask(all_eval_points, all_eval_pred, all_eval_target), [0, 1, 2, 3, 4])[0]
            precision = tp / (tp + fp)
            recall = tp / (tp + fp)
            f1 = 2 * (recall * precision) / (recall + precision)
            percentage_category_confusion = [round(tp / (tp + fn), 3), round(fp / (tn + fp), 3),
                                             round(fn / (tp + fn), 3), round(tn / (tn + fp), 3)]
            # Could do this with sklearn.metrics.confusion_matrix(normalise='true')

            wandb.log({'Validation/confusion_matrix': wandb.plot.confusion_matrix(probs=None, y_true=all_eval_target,
                                                                                  preds=all_eval_pred,
                                                                                  class_names=["keep", "discard"]),
                       'Validation/TP': tp, 'Validation/FP': fp,
                       'Validation/FN': fn, 'Validation/TN': tn,
                       'Validation/category-TP': percentage_category_confusion[0],
                       'Validation/category-FP': percentage_category_confusion[1],
                       'Validation/category-FN': percentage_category_confusion[2],
                       'Validation/category-TN': percentage_category_confusion[3],
                       'Validation/Precision': precision, 'Validation/Recall': recall, 'Validation/F1': f1},
                      commit=False)

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
                   'Validation/eval_point_avg_class_accuracy': eval_point_avg_class_accuracy}, commit=False)

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                str(l) + ' ' * (14 - 1), labelweights[l - 1],  # CHECK: pretty sure this reverse the weights
                total_correct_class[l] / float(total_iou_denominator_class[l]))  # refactor

        log_string(iou_per_class_str)
        log_string('Eval mean loss: %f' % eval_mean_loss)
        log_string('Eval accuracy: %f' % eval_point_accuracy)

        if mIoU >= best_iou:
            best_iou = mIoU
            log_string('Save best val_mIoU model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'class_avg_iou': mIoU,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saved best val_mIoU model.')
            wandb.save(savepath)
        log_string('Best mIoU: %f' % best_iou)
        return best_iou

    # HYPER PARAMETER
    # os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    experiment_dir, log_dir, checkpoints_dir = setup_logging_dir(config, '')
    logger = logging.getLogger("Model")
    setup_logger(logger, log_dir, config)
    log_string('PARAMETERS ...')
    log_string(config)
    setup_wandb_metrics()

    # Define constants
    DATA_PATH = Path(config["data_path"])
    NUM_CLASSES = 2

    NUM_POINTS = config["npoint"]
    BATCH_SIZE = config["batch_size"]
    BLOCK_SIZE = config["block_size"]

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = config["step_size"]

    SAVE_CURRENT_EPOCH_PREDS = False

    # Setup data_loaders and classifier
    TRAIN_DATASET, VAL_DATASET, train_data_loader, val_data_loader, weights = setup_data_loaders()
    classifier, criterion, optimizer, start_epoch = setup_model()

    # Training loop
    run_epoch = 0
    best_val_iou, best_train_iou = 0, 0
    if config["active_learning"]:
        # For AL we pass in the number of epochs to run AL for as config["epoch"].
        # Adding start epoch (from the pretrained model) offsets correctly
        if config["validate_only"]:
            start_epoch = config["epoch - 1"]
        else:
            # from start epoch train another K epochs (as given by config["epoch)
            config.update({"epoch": start_epoch + config["epoch"]}, allow_val_change=True)

    for epoch in range(start_epoch, config["epoch"]):
        log_string(f'**** Epoch {run_epoch + 1} ({epoch + 1}/{config["epoch"]}) ****')
        wandb.log({'epoch': epoch}, commit=False)
        lr, momentum = update_lr_momentum()

        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(train_data_loader)
        total_correct, total_seen, loss_sum = 0, 0, 0

        classifier = classifier.train()  # Set model to training mode

        if not config["validate_only"]:
            all_train_points, all_train_pred, all_train_target = [], [], []
            total_seen_class, total_correct_class, total_iou_denominator_class = [0, 0], [0, 0], [0, 0]

            for i, (points, target_labels) in tqdm(enumerate(train_data_loader), total=len(train_data_loader),
                                                   desc="Training"):
                optimizer.zero_grad()

                points = points.data.numpy()
                if config["augment_points"]: points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
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

                # Logging and visualisation and IoU
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_labels == l))  # How many times the label was in the batch
                    # How often the predicted label was correct in the batch
                    total_correct_class[l] += np.sum((pred_choice == l) & (batch_labels == l))
                    # Total predictions + Class occurrences (Union prediction of class (right or wrong) and actual class occurrences.)
                    total_iou_denominator_class[l] += np.sum(((pred_choice == l) | (batch_labels == l)))

                wandb.log({'Train/inner_epoch/loss_sum': loss_sum,
                           'Train/inner_epoch/accuracy_sum': total_correct / total_seen,
                           'Train/inner_epoch/loss': loss,
                           'Train/inner_epoch/accuracy': correct / len(batch_labels),
                           'Train/inner_epoch/class_ratio(percent_keeps)': total_seen_class[1] / sum(total_seen_class),
                           'Train/inner_epoch_step': (i + epoch * len(train_data_loader))}, commit=False)
                if config["log_merged_training_set"]:
                    if config["relative_point_coords"]:
                        all_train_points.append(np.array(points.transpose(1, 2).cpu())[:, :, [6, 7, 8, 3]])
                    else:
                        all_train_points.append(np.array(points.transpose(1, 2).cpu()))
                    all_train_pred.append(pred_choice.reshape(BATCH_SIZE, -1))
                    all_train_target.append(np.array(target_labels.cpu()).reshape(BATCH_SIZE, -1))
                # Visualise the first batch in every sample
                if config["log_first_batch_cloud"] and i == 0:
                    print(f"Visualising Epoch {epoch} Mini-Batch {i}")
                    visualise_batch(np.array(points.transpose(1, 2).cpu()), pred_choice.reshape(BATCH_SIZE, -1),
                                    np.array(target_labels.cpu()).reshape(BATCH_SIZE, -1), i, epoch, 'Train',
                                    pred_labels.exp().cpu().data.numpy(), config["log_merged_training_batches"])

            best_train_iou = post_training_logging_and_vis(all_train_points, all_train_pred, all_train_target,
                                                           best_train_iou)

            del all_train_points, all_train_pred, all_train_target

        with torch.no_grad():
            num_batches = len(val_data_loader)
            total_correct, total_seen, loss_sum = 0, 0, 0

            labelweights = np.zeros(2)  # only used for printing metrics
            total_seen_class, total_correct_class, total_iou_denominator_class = [0, 0], [0, 0], [0, 0]

            classifier = classifier.eval()
            all_eval_points, all_eval_pred, all_eval_target, all_eval_variance, all_eval_features = [], [], [], [], []

            repeats = config["validation_repeats"]
            if config["active_learning"] is True:
                log_string("Enabling dropout")
                enable_dropout(classifier)

            log_string(f'---- EPOCH {run_epoch + 1:03d} ({epoch + 1}/{config["epoch"]}) VALIDATION ----')
            for i, (points, target_labels) in tqdm(enumerate(val_data_loader), total=len(val_data_loader),
                                                   desc="Validation"):
                labelweights, total_correct, total_seen, loss_sum = validation_batch(BATCH_SIZE, NUM_CLASSES,
                                                                                     NUM_POINTS, all_eval_points,
                                                                                     all_eval_pred, all_eval_target,
                                                                                     config, classifier, criterion,
                                                                                     epoch, i, labelweights,
                                                                                     loss_sum, points,
                                                                                     target_labels, total_correct,
                                                                                     total_correct_class,
                                                                                     total_iou_denominator_class,
                                                                                     total_seen, total_seen_class,
                                                                                     val_data_loader, weights,
                                                                                     repeats, all_eval_variance,
                                                                                     all_eval_features)

            if config["log_merged_validation"]:
                all_eval_points = np.vstack(np.vstack(all_eval_points))
                all_eval_pred = np.hstack(np.vstack(all_eval_pred))
                all_eval_target = np.hstack(np.vstack(all_eval_target))
            if config["active_learning"]:
                all_eval_variance = np.hstack(all_eval_variance)
                all_eval_features = np.vstack(np.vstack(all_eval_features))
            best_val_iou = post_validation_logging_and_vis(all_eval_points, all_eval_pred, all_eval_target,
                                                           labelweights,
                                                           best_val_iou, all_eval_variance, all_eval_features)

        run_epoch += 1
        wandb.log({})
        exit(0)

    log_string("Finished")


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def np_mode_row(ar):
    """
    Return the row-wise most commmon element
    :param ar: a numpy array
    :return: an array of the most common element in each row
    """
    _min = np.min(ar)
    adjusted = False
    if _min < 0:
        ar = ar - _min
        adjusted = True
    ans = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=ar)
    if adjusted:
        ans = ans + _min
    return ans


def predictive_entropy(predictions):
    """
    Models how "surprised" the model is
    https://towardsdatascience.com/2-easy-ways-to-measure-your-image-classification-models-uncertainty-1c489fefaec8
    They are expecting it as a 10-class prediction stcked 5 times: (5,10)
    We have a 2 class prediction stacked 5 times (but with N batches); (5, N, 2)
    :param predictions:
    :return:
    """
    EPSILON = 1e-10
    pred_entropy = -np.sum(np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0) + EPSILON), axis=-1)
    return predictive_entropy


def variation_ratio(arr):
    """
    A measure of how "spread" the distribution is around the mode. In the binary case (ours) it is bounded by [0,0.5]
    Similar to simply taking the variance.
    :param arr:
    :return:
    """
    raise NotImplementedError


def binary_row_mode(arr):
    """
    Computes the column-wise mode of arr binary array
    :param arr: a 2d binary array
    :return:
    """
    out = np.empty(arr.shape[0], dtype=np.bool_)
    for i in range(len(out)):
        # out[i] = mode1d(arr[:,i])
        size = arr[i].size
        count = np.sum(arr[i])
        count = np.left_shift(count, 1)
        out[i] = count // size >= 1
    return out


# @profile
def validation_batch(BATCH_SIZE, NUM_CLASSES, NUM_POINTS, all_eval_points, all_eval_pred, all_eval_target, config,
                     classifier, criterion, epoch, i, labelweights, loss_sum, points, target_labels, total_correct,
                     total_correct_class, total_iou_denominator_class, total_seen, total_seen_class, val_data_loader,
                     weights, repeats=1, all_eval_variance=None, all_eval_features=None):
    if torch.is_tensor(points):
        points = points.data.numpy()
    points = torch.Tensor(points)
    if torch.is_tensor(target_labels):
        target_labels = target_labels.data.numpy()
    target_labels = torch.Tensor(target_labels)
    points, target_labels = points.float().cuda(), target_labels.long().cuda()
    points = points.transpose(2, 1)
    pred_logits, trans_feat, pred_choice = [], None, None

    # Run MC Dropout to get T sets of predictions.
    pred_logits, trans_feat = classifier(points, repeats != 1, repeats)  # trans_feat = high dimensional feature space
    if repeats != 1:
        # from scipy.stats import mode
        pred_logits = torch.stack(pred_logits)
        pred_logits = pred_logits.contiguous().view(repeats, -1, 2)
        pred_choice = pred_logits.cpu().data.max(2)[1].numpy().astype('int8')  # (repeats,N) labels
        pred_variances = pred_choice.var(axis=0)  # get the variance of the ensemble predictions
        pred_variances = pred_variances.reshape(target_labels.shape)
        # Get the sum(variance) over each batch-cell (grid cells may be split into many batch-cells)
        pred_variances = pred_variances.sum(axis=1).astype('float32')
        all_eval_variance.append(pred_variances)

        from scipy.stats import mode
        pred_choice = mode(pred_choice)
        pred_choice = pred_choice[0].ravel()  # N average labels
        # pred_choice = binary_row_mode(pred_choice)

        pred_logits = pred_logits[0]  # Just take one of them we only need it to calculate the loss
        all_eval_features.append(np.mean(trans_feat.cpu().numpy(), axis=-1))
    else:
        pred_logits = pred_logits.contiguous().view(-1, 2)
        pred_choice = pred_logits.cpu().data.max(1)[1].numpy()

    # CHECK whats happening here?
    batch_labels = target_labels.view(-1, 1)[:, 0].cpu().data.numpy()
    target_labels = target_labels.view(-1, 1)[:, 0]
    loss = criterion(pred_logits, target_labels, trans_feat, weights)
    correct = np.sum(pred_choice == batch_labels)
    total_correct += correct
    total_seen += (BATCH_SIZE * NUM_POINTS)
    loss_sum += loss
    tmp, _ = np.histogram(batch_labels, range(NUM_CLASSES + 1))
    labelweights += tmp
    wandb.log({'Validation/inner_epoch/loss_sum': loss_sum,
               'Validation/inner_epoch/accuracy_sum': total_correct / total_seen,
               'Validation/inner_epoch/loss': loss,
               'Validation/inner_epoch/accuracy': correct / len(batch_labels),
               'Validation/inner_epoch_step': (i + epoch * len(val_data_loader))}, commit=False)
    # Logging and visualisation and IoU
    for l in range(NUM_CLASSES):
        total_seen_class[l] += np.sum((batch_labels == l))  # How many times the label was in the batch
        # How often the predicted label was correct in the batch
        total_correct_class[l] += np.sum((pred_choice == l) & (batch_labels == l))
        # Class occurrences + total predictions (Union prediction of class (right or wrong) and actual class occurrences.)
        total_iou_denominator_class[l] += np.sum(((pred_choice == l) | (batch_labels == l)))
    if config["log_merged_validation"]:
        if config["relative_point_coords"]:
            all_eval_points.append(
                np.array(points.transpose(1, 2).cpu())[:, :, [6, 7, 8, 3]])  # Get the normalized xyzI
        else:
            all_eval_points.append(np.array(points.transpose(1, 2).cpu()))
        all_eval_pred.append(pred_choice.reshape(points.shape[0], -1))
        all_eval_target.append(np.array(target_labels.cpu()).astype('int8').reshape(points.shape[0], -1))
    if config["log_first_batch_cloud"] and i == 0:
        visualise_batch(np.array(points.transpose(1, 2).cpu()),
                        pred_choice.reshape(points.shape[0], -1),
                        np.array(target_labels.cpu()).reshape(points.shape[0], -1), i, epoch,
                        "Validation",
                        pred_logits.exp().cpu().numpy(), merged=config["log_merged_validation"])
    return labelweights, total_correct, total_seen, loss_sum


if __name__ == '__main__':
    args = parse_args()
    # os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="Masters", config=args, resume=False, group="50%Validation",
               name=f"{'-'.join(args.data_path.split('/')[-2:])}-50%Validation",
               notes="")
    wandb.run.log_code(".")
    main(wandb.config)
    wandb.finish()
#     Tested with local points => bad results
#     Tested as normal with higher WD 1e-2, => good results but can't resolve the fences and does less well on training set
#     Tested as normal with medium WD 1e-3
#     Tested with local points and higher WD 1e-2
#     as normal with adamW,
#     with local points and adamW with higher WD

#  Also want to test augmenting the points by rotating around the axis
#  Can also try forcing even classes
