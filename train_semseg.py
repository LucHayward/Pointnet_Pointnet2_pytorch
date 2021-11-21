"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os

from Visualisation_utils import visualise_batch, visualise_prediction, turbo_colormap_data
from data_utils.S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
import pptk

import wandb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--num_classes', type=int, default=13, help='number of class_labels [default: 13]')
    parser.add_argument('--block_size', default=1.0, type=float, help='column size for sampling (tbc)')
    parser.add_argument('--data_path', default=None,
                        help='If data path needs to change, set it here. Should point to data root')

    # New arguments
    parser.add_argument('--log_first_batch_cloud', action='store_true', help='Log the pointclouds that get sampled')
    parser.add_argument('--validate_only', action='store_true', help='Skip training and only run the validation step')
    parser.add_argument('--augment_points', action='store_true', help='Augment pointcloud (currently by rotation)')
    parser.add_argument('--log_merged_validation', action='store_true', help='Log the merged validation pointclouds')
    parser.add_argument('--log_merged_training_batches', action='store_true',
                        help='When logging training batch visualisations, merge batches in global coordinate space before visualising.')
    parser.add_argument('--log_merged_training_set', action='store_true',
                        help='merge all teh points used during training into one visualisation')
    parser.add_argument('--force_bn', action='store_true', help='Force the BatchNorm layers to be on during evaluation')
    parser.add_argument('--test_sample_rate', default=1.0, type=float, help='How much to oversample the test set by')
    parser.add_argument('--shuffle_training_data', action='store_true', help='Shuffle the training data loader')

    # Exposing new HParams
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


def set_bn_training(model, v):
    """Sets the models BatchNorm lauers to True"""
    if 'BatchNorm' in model._get_name():
        model.training = v
    for module in model.children():
        set_bn_training(module, v)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    if args.num_classes == 2:
        global classes, class2label, seg_classes, seg_label_to_cat
        classes = ['keep', 'discard']
        class2label = {cls: i for i, cls in enumerate(classes)}
        seg_classes = class2label
        seg_label_to_cat = {}
        for i, cat in enumerate(seg_classes.keys()):
            seg_label_to_cat[i] = cat

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
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

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/s3dis/stanford_indoor3d/'
    if args.data_path is not None:
        root = f'data/s3dis/{args.data_path}'
    NUM_CLASSES = args.num_classes
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area,
                                 block_size=args.block_size, sample_rate=1.0, transform=None, num_classes=NUM_CLASSES)
    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area,
                                block_size=1.0, sample_rate=args.test_sample_rate, transform=None,
                                num_classes=NUM_CLASSES)
    # DEBUG
    # import pptk
    # all_points = [p for sublist in TRAIN_DATASET.room_points for p in sublist]
    # all_labels = [p for sublist in TRAIN_DATASET.room_labels for p in sublist]
    # all_labels = np.array(all_labels)
    # all_points = np.array(all_points)
    # current_points, current_labels = TRAIN_DATASET.room_points[0], TRAIN_DATASET.room_labels[0]
    # v = pptk.viewer(current_points[:,:3], current_points[:,:3], current_labels)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE,
                                                  shuffle=args.shuffle_training_data, num_workers=0,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                                 pin_memory=True,
                                                 drop_last=(True if len(TEST_DATASET) // BATCH_SIZE else False))
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    wandb.config.update({'num_training_data': len(TRAIN_DATASET),
                         'num_test-data': len(TEST_DATASET)})

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    wandb.watch(classifier, criterion, log="all", log_freq=10)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

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

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    training_examples_seen = 0
    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
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

        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()  # Set model to training mode

        if not args.validate_only:
            # TODO: Compare labeleweights (hist(batch_labels)) and maybe log this?
            # TODO: maybe even start logging the training vs prediction images?
            all_train_points = []
            all_train_pred = []
            all_train_target = []
            for i, (points, target, room_idx) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader),
                                                      smoothing=0.9,
                                                      desc="Training"):
                optimizer.zero_grad()
                # Points: Global XYZ, IGB/255, XYZ/max(room_XYZ)
                points = points.data.numpy()
                # if args.log_first_batch_cloud:
                # Log points and their places
                # v = pptk.viewer(np.vstack(points)[:, :3], np.hstack(target), np.repeat(room_idx, 4096),
                #                 np.arange(20).repeat(4096))
                # v.color_map(turbo_colormap_data)
                # print(np.asarray(np.unique(room_idx, return_counts=True)).T)
                # CHECK Should we be augmenting? I think it helps the model be more robust
                if args.augment_points: points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)  # Convert points to num_batches * 9 * num_points, No idea why though

                seg_pred, trans_feat = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss.backward()
                optimizer.step()

                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                loss_sum += loss

                wandb.log({'Train/inner_epoch_loss_sum': loss_sum,
                           'Train/inner_epoch_accuracy_sum': total_correct / total_seen,
                           'Train/inner_epoch_loss': loss,
                           'Train/inner_epoch_accuracy': correct / len(batch_label),
                           'epoch': epoch,
                           'Train/inner_epoch_step': (i + epoch * len(trainDataLoader))})

                if args.log_merged_training_set and args.num_classes == 2:
                    all_train_points.append(np.array(points.transpose(1, 2).cpu()))
                    all_train_pred.append(pred_choice.reshape(args.batch_size, -1))
                    all_train_target.append(np.array(target.cpu()).reshape(args.batch_size, -1))

                if args.log_first_batch_cloud and i == 0:  # Visualise the first batch in every sample
                    print(f"Visualising Epoch {epoch} Mini-Batch {i}")
                    visualise_batch(np.array(points.transpose(1, 2).cpu()), pred_choice.reshape(args.batch_size, -1),
                                    np.array(target.cpu()).reshape(args.batch_size, -1), i, epoch, 'Train',
                                    seg_pred.exp().cpu().data.numpy(), args.log_merged_training_batches)

            if args.log_merged_training_set and args.num_classes == 2:
                all_train_points = np.vstack(np.vstack(all_train_points))
                all_train_pred = np.hstack(np.vstack(all_train_pred))
                all_train_target = np.hstack(np.vstack(all_train_target))
                _, unique_indices, unique_counts = np.unique(all_train_points[:, :3], axis=0, return_index=True,
                                                             return_counts=True)
                num_unique_points = len(unique_indices)
                total_training_points = np.vstack(TRAIN_DATASET.room_points).shape[0]
                print(
                    f"Unique points: {num_unique_points}/{total_training_points} ({num_unique_points * 100 // total_training_points}%)")
                visualise_prediction(all_train_points[:, :3], all_train_pred,
                                     all_train_target, epoch,
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

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)  # only used for printing metrics
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_denominator_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            if args.force_bn:
                set_bn_training(classifier, True)

            all_eval_points = []
            all_eval_pred = []
            all_eval_target = []
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target, room_idx) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_choice = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_choice = np.argmax(pred_choice, 2)
                correct = np.sum((pred_choice == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))  # How many times the label was in the batch
                    total_correct_class[l] += np.sum(
                        (pred_choice == l) & (
                                batch_label == l))  # How often the predicted label was correct in the batch
                    total_iou_denominator_class[l] += np.sum(((pred_choice == l) | (
                            batch_label == l)))  # Class occurrences + total predictions (Union prediction of class (right or wrong) and actual class occurrences.)

                if args.log_merged_validation and args.num_classes == 2:
                    all_eval_points.append(np.array(points.transpose(1, 2).cpu()))
                    all_eval_pred.append(pred_choice.reshape(points.shape[0], -1))
                    all_eval_target.append(np.array(target.cpu()).reshape(points.shape[0], -1))

                if args.log_first_batch_cloud and i == 0:
                    visualise_batch(np.array(points.transpose(1, 2).cpu()), pred_choice.reshape(points.shape[0], -1),
                                    np.array(target.cpu()).reshape(points.shape[0], -1), i, epoch, "Validation",
                                    seg_pred.exp().cpu().numpy(), merged=args.log_merged_validation)

            if args.log_merged_validation and args.num_classes == 2:
                all_eval_points = np.vstack(np.vstack(all_eval_points))
                all_eval_pred = np.hstack(np.vstack(all_eval_pred))
                all_eval_target = np.hstack(np.vstack(all_eval_target))
                _, unique_indices = np.unique(all_eval_points[:, :3], axis=0, return_index=True)
                num_unique_points = len(unique_indices)
                total_eval_points = np.vstack(TEST_DATASET.room_points).shape[0]
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
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
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
        global_epoch += 1
        wandb.log({})


if __name__ == '__main__':
    args = parse_args()
    config = {'grid_shape_original': (10, 10,), 'data_split': {'training': 10, 'validation': 2}}  # TODO: Dynamically
    config = {}
    config.update(args.__dict__)
    # os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="PointNet2-Pytorch",
               config=config, name='SongoMnara-Double-Points', resume=True)
    wandb.run.log_code(".")
    main(args)
    wandb.finish()

