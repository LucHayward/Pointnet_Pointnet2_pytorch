"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
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

import wandb
import pptk
from tabulate import tabulate
from line_profiler_pycharm import profile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

turbo_colormap_data = [[0.18995, 0.07176, 0.23217], [0.19483, 0.08339, 0.26149], [0.19956, 0.09498, 0.29024],
                       [0.20415, 0.10652, 0.31844], [0.20860, 0.11802, 0.34607], [0.21291, 0.12947, 0.37314],
                       [0.21708, 0.14087, 0.39964], [0.22111, 0.15223, 0.42558], [0.22500, 0.16354, 0.45096],
                       [0.22875, 0.17481, 0.47578], [0.23236, 0.18603, 0.50004], [0.23582, 0.19720, 0.52373],
                       [0.23915, 0.20833, 0.54686], [0.24234, 0.21941, 0.56942], [0.24539, 0.23044, 0.59142],
                       [0.24830, 0.24143, 0.61286], [0.25107, 0.25237, 0.63374], [0.25369, 0.26327, 0.65406],
                       [0.25618, 0.27412, 0.67381], [0.25853, 0.28492, 0.69300], [0.26074, 0.29568, 0.71162],
                       [0.26280, 0.30639, 0.72968], [0.26473, 0.31706, 0.74718], [0.26652, 0.32768, 0.76412],
                       [0.26816, 0.33825, 0.78050], [0.26967, 0.34878, 0.79631], [0.27103, 0.35926, 0.81156],
                       [0.27226, 0.36970, 0.82624], [0.27334, 0.38008, 0.84037], [0.27429, 0.39043, 0.85393],
                       [0.27509, 0.40072, 0.86692], [0.27576, 0.41097, 0.87936], [0.27628, 0.42118, 0.89123],
                       [0.27667, 0.43134, 0.90254], [0.27691, 0.44145, 0.91328], [0.27701, 0.45152, 0.92347],
                       [0.27698, 0.46153, 0.93309], [0.27680, 0.47151, 0.94214], [0.27648, 0.48144, 0.95064],
                       [0.27603, 0.49132, 0.95857], [0.27543, 0.50115, 0.96594], [0.27469, 0.51094, 0.97275],
                       [0.27381, 0.52069, 0.97899], [0.27273, 0.53040, 0.98461], [0.27106, 0.54015, 0.98930],
                       [0.26878, 0.54995, 0.99303], [0.26592, 0.55979, 0.99583], [0.26252, 0.56967, 0.99773],
                       [0.25862, 0.57958, 0.99876], [0.25425, 0.58950, 0.99896], [0.24946, 0.59943, 0.99835],
                       [0.24427, 0.60937, 0.99697], [0.23874, 0.61931, 0.99485], [0.23288, 0.62923, 0.99202],
                       [0.22676, 0.63913, 0.98851], [0.22039, 0.64901, 0.98436], [0.21382, 0.65886, 0.97959],
                       [0.20708, 0.66866, 0.97423], [0.20021, 0.67842, 0.96833], [0.19326, 0.68812, 0.96190],
                       [0.18625, 0.69775, 0.95498], [0.17923, 0.70732, 0.94761], [0.17223, 0.71680, 0.93981],
                       [0.16529, 0.72620, 0.93161], [0.15844, 0.73551, 0.92305], [0.15173, 0.74472, 0.91416],
                       [0.14519, 0.75381, 0.90496], [0.13886, 0.76279, 0.89550], [0.13278, 0.77165, 0.88580],
                       [0.12698, 0.78037, 0.87590], [0.12151, 0.78896, 0.86581], [0.11639, 0.79740, 0.85559],
                       [0.11167, 0.80569, 0.84525], [0.10738, 0.81381, 0.83484], [0.10357, 0.82177, 0.82437],
                       [0.10026, 0.82955, 0.81389], [0.09750, 0.83714, 0.80342], [0.09532, 0.84455, 0.79299],
                       [0.09377, 0.85175, 0.78264], [0.09287, 0.85875, 0.77240], [0.09267, 0.86554, 0.76230],
                       [0.09320, 0.87211, 0.75237], [0.09451, 0.87844, 0.74265], [0.09662, 0.88454, 0.73316],
                       [0.09958, 0.89040, 0.72393], [0.10342, 0.89600, 0.71500], [0.10815, 0.90142, 0.70599],
                       [0.11374, 0.90673, 0.69651], [0.12014, 0.91193, 0.68660], [0.12733, 0.91701, 0.67627],
                       [0.13526, 0.92197, 0.66556], [0.14391, 0.92680, 0.65448], [0.15323, 0.93151, 0.64308],
                       [0.16319, 0.93609, 0.63137], [0.17377, 0.94053, 0.61938], [0.18491, 0.94484, 0.60713],
                       [0.19659, 0.94901, 0.59466], [0.20877, 0.95304, 0.58199], [0.22142, 0.95692, 0.56914],
                       [0.23449, 0.96065, 0.55614], [0.24797, 0.96423, 0.54303], [0.26180, 0.96765, 0.52981],
                       [0.27597, 0.97092, 0.51653], [0.29042, 0.97403, 0.50321], [0.30513, 0.97697, 0.48987],
                       [0.32006, 0.97974, 0.47654], [0.33517, 0.98234, 0.46325], [0.35043, 0.98477, 0.45002],
                       [0.36581, 0.98702, 0.43688], [0.38127, 0.98909, 0.42386], [0.39678, 0.99098, 0.41098],
                       [0.41229, 0.99268, 0.39826], [0.42778, 0.99419, 0.38575], [0.44321, 0.99551, 0.37345],
                       [0.45854, 0.99663, 0.36140], [0.47375, 0.99755, 0.34963], [0.48879, 0.99828, 0.33816],
                       [0.50362, 0.99879, 0.32701], [0.51822, 0.99910, 0.31622], [0.53255, 0.99919, 0.30581],
                       [0.54658, 0.99907, 0.29581], [0.56026, 0.99873, 0.28623], [0.57357, 0.99817, 0.27712],
                       [0.58646, 0.99739, 0.26849], [0.59891, 0.99638, 0.26038], [0.61088, 0.99514, 0.25280],
                       [0.62233, 0.99366, 0.24579], [0.63323, 0.99195, 0.23937], [0.64362, 0.98999, 0.23356],
                       [0.65394, 0.98775, 0.22835], [0.66428, 0.98524, 0.22370], [0.67462, 0.98246, 0.21960],
                       [0.68494, 0.97941, 0.21602], [0.69525, 0.97610, 0.21294], [0.70553, 0.97255, 0.21032],
                       [0.71577, 0.96875, 0.20815], [0.72596, 0.96470, 0.20640], [0.73610, 0.96043, 0.20504],
                       [0.74617, 0.95593, 0.20406], [0.75617, 0.95121, 0.20343], [0.76608, 0.94627, 0.20311],
                       [0.77591, 0.94113, 0.20310], [0.78563, 0.93579, 0.20336], [0.79524, 0.93025, 0.20386],
                       [0.80473, 0.92452, 0.20459], [0.81410, 0.91861, 0.20552], [0.82333, 0.91253, 0.20663],
                       [0.83241, 0.90627, 0.20788], [0.84133, 0.89986, 0.20926], [0.85010, 0.89328, 0.21074],
                       [0.85868, 0.88655, 0.21230], [0.86709, 0.87968, 0.21391], [0.87530, 0.87267, 0.21555],
                       [0.88331, 0.86553, 0.21719], [0.89112, 0.85826, 0.21880], [0.89870, 0.85087, 0.22038],
                       [0.90605, 0.84337, 0.22188], [0.91317, 0.83576, 0.22328], [0.92004, 0.82806, 0.22456],
                       [0.92666, 0.82025, 0.22570], [0.93301, 0.81236, 0.22667], [0.93909, 0.80439, 0.22744],
                       [0.94489, 0.79634, 0.22800], [0.95039, 0.78823, 0.22831], [0.95560, 0.78005, 0.22836],
                       [0.96049, 0.77181, 0.22811], [0.96507, 0.76352, 0.22754], [0.96931, 0.75519, 0.22663],
                       [0.97323, 0.74682, 0.22536], [0.97679, 0.73842, 0.22369], [0.98000, 0.73000, 0.22161],
                       [0.98289, 0.72140, 0.21918], [0.98549, 0.71250, 0.21650], [0.98781, 0.70330, 0.21358],
                       [0.98986, 0.69382, 0.21043], [0.99163, 0.68408, 0.20706], [0.99314, 0.67408, 0.20348],
                       [0.99438, 0.66386, 0.19971], [0.99535, 0.65341, 0.19577], [0.99607, 0.64277, 0.19165],
                       [0.99654, 0.63193, 0.18738], [0.99675, 0.62093, 0.18297], [0.99672, 0.60977, 0.17842],
                       [0.99644, 0.59846, 0.17376], [0.99593, 0.58703, 0.16899], [0.99517, 0.57549, 0.16412],
                       [0.99419, 0.56386, 0.15918], [0.99297, 0.55214, 0.15417], [0.99153, 0.54036, 0.14910],
                       [0.98987, 0.52854, 0.14398], [0.98799, 0.51667, 0.13883], [0.98590, 0.50479, 0.13367],
                       [0.98360, 0.49291, 0.12849], [0.98108, 0.48104, 0.12332], [0.97837, 0.46920, 0.11817],
                       [0.97545, 0.45740, 0.11305], [0.97234, 0.44565, 0.10797], [0.96904, 0.43399, 0.10294],
                       [0.96555, 0.42241, 0.09798], [0.96187, 0.41093, 0.09310], [0.95801, 0.39958, 0.08831],
                       [0.95398, 0.38836, 0.08362], [0.94977, 0.37729, 0.07905], [0.94538, 0.36638, 0.07461],
                       [0.94084, 0.35566, 0.07031], [0.93612, 0.34513, 0.06616], [0.93125, 0.33482, 0.06218],
                       [0.92623, 0.32473, 0.05837], [0.92105, 0.31489, 0.05475], [0.91572, 0.30530, 0.05134],
                       [0.91024, 0.29599, 0.04814], [0.90463, 0.28696, 0.04516], [0.89888, 0.27824, 0.04243],
                       [0.89298, 0.26981, 0.03993], [0.88691, 0.26152, 0.03753], [0.88066, 0.25334, 0.03521],
                       [0.87422, 0.24526, 0.03297], [0.86760, 0.23730, 0.03082], [0.86079, 0.22945, 0.02875],
                       [0.85380, 0.22170, 0.02677], [0.84662, 0.21407, 0.02487], [0.83926, 0.20654, 0.02305],
                       [0.83172, 0.19912, 0.02131], [0.82399, 0.19182, 0.01966], [0.81608, 0.18462, 0.01809],
                       [0.80799, 0.17753, 0.01660], [0.79971, 0.17055, 0.01520], [0.79125, 0.16368, 0.01387],
                       [0.78260, 0.15693, 0.01264], [0.77377, 0.15028, 0.01148], [0.76476, 0.14374, 0.01041],
                       [0.75556, 0.13731, 0.00942], [0.74617, 0.13098, 0.00851], [0.73661, 0.12477, 0.00769],
                       [0.72686, 0.11867, 0.00695], [0.71692, 0.11268, 0.00629], [0.70680, 0.10680, 0.00571],
                       [0.69650, 0.10102, 0.00522], [0.68602, 0.09536, 0.00481], [0.67535, 0.08980, 0.00449],
                       [0.66449, 0.08436, 0.00424], [0.65345, 0.07902, 0.00408], [0.64223, 0.07380, 0.00401],
                       [0.63082, 0.06868, 0.00401], [0.61923, 0.06367, 0.00410], [0.60746, 0.05878, 0.00427],
                       [0.59550, 0.05399, 0.00453], [0.58336, 0.04931, 0.00486], [0.57103, 0.04474, 0.00529],
                       [0.55852, 0.04028, 0.00579], [0.54583, 0.03593, 0.00638], [0.53295, 0.03169, 0.00705],
                       [0.51989, 0.02756, 0.00780], [0.50664, 0.02354, 0.00863], [0.49321, 0.01963, 0.00955],
                       [0.47960, 0.01583, 0.01055]]

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


def convert_class_to_rgb255(class_num, max_class):
    if class_num == 0:
        return np.multiply(turbo_colormap_data[0], 255)
    elif class_num == max_class:
        return np.multiply(turbo_colormap_data[-1], 255)
    else:
        return np.multiply(turbo_colormap_data[len(turbo_colormap_data) // max_class * class_num], 255)


# 1790+324+27+1955
# @profile
def visualise_batch(points, pred, target_labels, batch_num, epoch, data_split, experiment_dir=Path(''),
                    confidences=False, merged=False):
    if merged:
        visualise_prediction(np.vstack(points)[:, :3], np.hstack(pred), np.hstack(target_labels), epoch, data_split,
                             batch_num, confidences=confidences, wandb_section='Visualise-Batch')
    else:
        for idx, points_batch in tqdm(enumerate(points), total=len(points), desc="visualise_batch"):
            # print(f'\nEpoch {epoch} batch {batch_num} sample {idx}:')
            visualise_prediction(points_batch[:, :3], pred[idx], target_labels[idx], epoch, data_split, batch_num,
                                 wandb_section='Visualise-Batch')


def visualise_prediction(points, pred, target_labels, epoch, data_split, batch_num=None, confidences=None,
                         wandb_section=None):
    confusion_mask = np.zeros(len(points), dtype=int)  # pptk/wandb
    confusion_mask[(pred == target_labels) & (pred == 1)] = 3  # tp (red)
    confusion_mask[(pred != target_labels) & (pred == 0)] = 2  # fn (yellow)
    confusion_mask[(pred != target_labels) & (pred == 1)] = 1  # fp (green)
    confusion_mask[(pred == target_labels) & (pred == 0)] = 0  # tn (purple)

    #         confusion_mask[(pred[idx] != target_labels[idx]) & (pred[idx] == 1)] = 3  # fp (red)
    #         confusion_mask[(pred[idx] != target_labels[idx]) & (pred[idx] == 0)] = 2  # fn (yellow)
    #         confusion_mask[(pred[idx] == target_labels[idx]) & (pred[idx] == 1)] = 1  # tp (green)
    #         confusion_mask[(pred[idx] == target_labels[idx]) & (pred[idx] == 0)] = 0  # tn (purple)

    confusion_mask_rgb255 = np.array([convert_class_to_rgb255(i, 3) for i in confusion_mask])
    pred_rgb255 = np.array([convert_class_to_rgb255(i, 1) for i in pred])
    target_labels_rgb255 = np.array([convert_class_to_rgb255(i, 1) for i in target_labels])
    if confidences is not None:
        confidences_rgb255 = np.array([])

    confusion_matrix_data = np.histogram(confusion_mask, [0, 1, 2, 3, 4])
    accuracy = np.sum(pred == target_labels) / len(pred)
    # Precision = of all the positive predications how many are correct (high Precision = low FP)
    precision = confusion_matrix_data[0][3] / (confusion_matrix_data[0][3] + confusion_matrix_data[0][1])
    recall = confusion_matrix_data[0][3] / (confusion_matrix_data[0][3] + confusion_matrix_data[0][2])  # sensitivity
    f1 = 2 * (recall * precision) / (recall + precision)

    # TODO: Visualise confidence
    v = pptk.viewer(points)
    v.color_map(turbo_colormap_data)
    v.set(point_size=0.01, lookat=np.mean(points, axis=0), r=20, phi=.9, theta=0.4)
    # save_path = experiment_dir / f'media/pointclouds'
    # if not save_path.exists():
    #     save_path.mkdir(parents=True)
    # v.attributes(pred[idx])
    # v.capture(save_path / f'E{epoch}B{batch_num}Predicted.png')
    # v.attributes(target_labels[idx])
    # v.capture(save_path / f'E{epoch}B{batch_num}Target.png')
    # v.attributes(confusion_mask)
    # v.capture(save_path / f'E{epoch}B{batch_num}Comparison.png')

    if confidences is not None:
        # Could do this by applying it as a label or as a alpha mask
        # Confidence of prediction intervals histogram
        print(print(np.histogram(confidences.max(1).round(1), np.linspace(0.5, 1,6))))
        v.attributes(pred, target_labels, confusion_mask, np.hstack((pred_rgb255 / 255, confidences.max(1).round(1)[:,None])))
        v.attributes(pred, target_labels, confusion_mask, np.hstack((confusion_mask_rgb255 / 255, confidences.max(1).round(1)[:,None])))
    else:
        v.attributes(pred, target_labels, confusion_mask)

    print(tabulate([['Pred 1', confusion_matrix_data[0][3], confusion_matrix_data[0][1]],
                    ['Pred 0', confusion_matrix_data[0][2], confusion_matrix_data[0][0]]],
                   headers=['', 'Actual 1', 'Actual 0']))
    print(tabulate([['Pred 1', (confusion_matrix_data[0][3]/len(target_labels)).__round__(3), (confusion_matrix_data[0][1]/len(target_labels)).__round__(3)],
                    ['Pred 0', (confusion_matrix_data[0][2]/len(target_labels)).__round__(3), (confusion_matrix_data[0][0]/len(target_labels)).__round__(3)]],
                   headers=['', 'Actual 1', 'Actual 0']))
    print(f'Accuracy: {accuracy}\n'
          f'Recall: {recall}\n'
          f'Precision: {precision}\n'
          f'f1: {f1}\n'
          f'Distribution (0:1): {(confusion_matrix_data[0][0] + confusion_matrix_data[0][1]) / len(target_labels):.2f}:'
          f'{(confusion_matrix_data[0][2] + confusion_matrix_data[0][3]) / len(target_labels):.2f}')

    if batch_num is not None:
        wandb.log({
            f"{(wandb_section + '/') if wandb_section is not None else ''}{data_split}/pointcloud-ground-truth-and-prediction": {
                'batch': batch_num}},
            commit=False)
    if confidences is not None:
        pass
        # wandb.log({
        #     f"{(wandb_section + '/') if wandb_section is not None else ''}{data_split}/pointcloud-ground-truth-and-prediction": {
        #         "pointcloud": {
        #             "confidence": wandb.Object3D(np.hstack((points, confidences_rgb255)))
        #
        #         }}}, commit=False)
    wandb.log({
        f"{(wandb_section + '/') if wandb_section is not None else ''}{data_split}/pointcloud-ground-truth-and-prediction": {
            "epoch": epoch,
            # "batch": batch_num,
            # "epoch*batch*sample": max(epoch, 1) * max(batch_num, 1) * (idx + 1),
            "confusion-matrix": {
                "histogram": wandb.Histogram(np_histogram=confusion_matrix_data),
                "true-positive": confusion_matrix_data[0][3],
                "false-positive": confusion_matrix_data[0][1],
                "true-negative": confusion_matrix_data[0][0],
                "false-negative": confusion_matrix_data[0][2]
            },
            'accuracy': accuracy,
            "pointcloud": {
                "confusion-cloud": wandb.Object3D(np.hstack((points, confusion_mask_rgb255))),
                "ground-truth": wandb.Object3D(np.hstack((points, target_labels_rgb255))),
                "prediction": wandb.Object3D(np.hstack((points, pred_rgb255)))
            }
        }})


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
    parser.add_argument('--log_clouds', action='store_true', help='Log the pointclouds that get sampled')
    parser.add_argument('--validate_only', action='store_true', help='Skip training and only run the validation step')
    parser.add_argument('--no_augment_points', action='store_false', help='Augment pointcloud (currently by rotation)')
    parser.add_argument('--log_merged_validation', action='store_true', help='Log the merged validation pointclouds')
    parser.add_argument('--log_merged_training_batches', action='store_true',
                        help='When logging training batch visualisations, merge batches in global coordinate space before visualising.')
    parser.add_argument('--force_bn', action='store_true', help='Force the BatchNorm layers to be on during evaluation')

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
                                block_size=1.0, sample_rate=1.0, transform=None, num_classes=NUM_CLASSES)
    # DEBUG
    # import pptk
    # all_points = [p for sublist in TRAIN_DATASET.room_points for p in sublist]
    # all_labels = [p for sublist in TRAIN_DATASET.room_labels for p in sublist]
    # all_labels = np.array(all_labels)
    # all_points = np.array(all_points)
    # current_points, current_labels = TRAIN_DATASET.room_points[0], TRAIN_DATASET.room_labels[0]
    # v = pptk.viewer(current_points[:,:3], current_points[:,:3], current_labels)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                                 pin_memory=True, drop_last=True)
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
            for i, (points, target, room_idx) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader),
                                                      smoothing=0.9,
                                                      desc="Training"):
                optimizer.zero_grad()
                # Points: Global XYZ, IGB/255, XYZ/max(room_XYZ)
                points = points.data.numpy()
                # if args.log_clouds:
                    # Log points and their places
                    # v = pptk.viewer(np.vstack(points)[:, :3], np.hstack(target), np.repeat(room_idx, 4096),
                    #                 np.arange(20).repeat(4096))
                    # v.color_map(turbo_colormap_data)
                    # print(np.asarray(np.unique(room_idx, return_counts=True)).T)
                # CHECK Should we be augmenting? I think it helps the model be more robust
                if not args.no_augment_points: points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
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
                # TODO: check this
                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                loss_sum += loss

                if args.log_clouds and i == 0:  # Visualise the first batch in every sample
                    visualise_batch(np.array(points.transpose(1, 2).cpu()),
                                    pred_choice.reshape(20, -1), np.array(target.cpu()).reshape(20, -1), i, epoch,
                                    'Train', experiment_dir, seg_pred.exp().cpu().data.numpy(),
                                    args.log_merged_training_batches)

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
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_denominator_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            if args.force_bn:
                set_bn_training(classifier, True)

            all_eval_points = []
            all_eval_target = []
            all_eval_pred = []
            all_eval_room_idx = []
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target, room_idx) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))  # How many times the label was in the batch
                    total_correct_class[l] += np.sum(
                        (pred_val == l) & (batch_label == l))  # How often the predicted label was correct in the batch
                    total_iou_denominator_class[l] += np.sum(((pred_val == l) | (
                            batch_label == l)))  # Class occurrences + total predictions (Union prediction of class (right or wrong) and actual class occurrences.)

                if args.log_merged_validation:
                    all_eval_points.append(np.array(points.transpose(1, 2).cpu()))
                    all_eval_pred.append(pred_val.reshape(20, -1))
                    all_eval_target.append(np.array(target.cpu()).reshape(20, -1))
                    all_eval_room_idx += room_idx
                if args.log_clouds and i == 0:
                    visualise_batch(np.array(points.transpose(1, 2).cpu()),
                                    pred_val.reshape(20, -1), np.array(target.cpu()).reshape(20, -1), i, epoch,
                                    "Validation", experiment_dir, seg_pred.exp().cpu().numpy(), merged=args.log_merged_validation)

            if args.log_merged_validation:
                stop_index = all_eval_room_idx.index(1) - 1
                start_index = 0
                for area in range(len(TEST_DATASET.room_points)):
                    pnts = np.concatenate(all_eval_points, axis=0)
                    pnts = pnts[start_index:stop_index]
                    pnts = pnts.reshape(-1, 9)

                    targets = np.concatenate(all_eval_target, axis=0)
                    targets = targets[start_index:stop_index]
                    targets = targets.reshape(-1)

                    preds = np.concatenate(all_eval_pred, axis=0)
                    preds = preds[start_index:stop_index]
                    preds = preds.reshape(-1)
                    visualise_prediction(pnts[:, 6:], preds, targets, epoch,
                                         "Validation", wandb_section="Visualise-Merged")

                    # TODO: second validation set isn't displaying correctly...

                    start_index = stop_index
                    stop_index = len(all_eval_room_idx)

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

            """
            ------- IoU --------
            class keep           weight: 0.253, IoU: 0.493
            class discard        weight: 0.747, IoU: 0.146
            """
            wandb.log({"Validation/IoU/": {
                "class": seg_label_to_cat[l],
                "weight": labelweights[l - 1],
                "IoU": total_correct_class[l] / float(total_iou_denominator_class[l]),
                "text": iou_per_class_str
            }
            }, commit=False)

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
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1
        wandb.log({})

def generate_points_on_sphere(r, n, offset=[0, 0, 0]):
    points = []
    alpha = 4.0 * np.pi * r * r / n
    d = np.sqrt(alpha)
    m_nu = int(np.round(np.pi / d))
    d_nu = np.pi / m_nu
    d_phi = alpha / d_nu
    count = 0
    for m in range(0, m_nu):
        nu = np.pi * (m + 0.5) / m_nu
        m_phi = int(np.round(2 * np.pi * np.sin(nu) / d_phi))
        for n in range(0, m_phi):
            phi = 2 * np.pi * n / m_phi
            xp = r * np.sin(nu) * np.cos(phi)
            yp = r * np.sin(nu) * np.sin(phi)
            zp = r * np.cos(nu)
            count = count + 1
            points.append([xp, yp, zp])

    return np.array(points) + offset

def generate_bounding_wireframe_points(min, max, number):
    points = [np.linspace(min, [max[0], min[1], min[2]], number), np.linspace(min, [min[0], max[1], min[2]], number),
              np.linspace(min, [min[0], min[1], max[2]], number),
              np.linspace([min[0], min[1], max[2]], [max[0], min[1], max[2]], number),
              np.linspace([min[0], min[1], max[2]], [min[0], max[1], max[2]], number),
              np.linspace([min[0], max[1], min[2]], [max[0], max[1], min[2]], number),
              np.linspace([min[0], max[1], min[2]], [min[0], max[1], max[2]], number),
              np.linspace([max[0], min[1], min[2]], [max[0], max[1], min[2]], number),
              np.linspace([max[0], min[1], min[2]], [max[0], min[1], max[2]], number),
              np.linspace(max, [min[0], max[1], max[2]], number), np.linspace(max, [max[0], min[1], max[2]], number),
              np.linspace(max, [max[0], max[1], min[2]], number)]

    points = np.vstack(points)
    colours = np.ones(points.shape)
    # vv = pptk.viewer(points, colours)
    # vv.set(point_size=0.01)
    return points, colours


def generate_bounding_cube(origin, size):
    return generate_bounding_wireframe_points(np.array(origin), np.array(origin) + size, 10 * size)


if __name__ == '__main__':
    args = parse_args()
    config = {'grid_shape_original': (10, 10,), 'data_split': {'training': 9, 'validation': 2}}
    config.update(args.__dict__)
    # os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="PointNet2-Pytorch",
               config=config)
    main(args)
    wandb.finish()
    ################
    #
    # args.__dict__.update({'npoint': 4096 * 2})
    # config = {'grid_shape_original': (10, 10,), 'data_split': {'training': 9, 'validation': 2}}
    # config.update(args.__dict__)
    # # os.environ["WANDB_MODE"] = "dryrun"
    # wandb.init(project="PointNet2-Pytorch",
    #            config=config, name="Church-Grid")
    # main(args)
