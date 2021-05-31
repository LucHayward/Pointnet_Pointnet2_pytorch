import os
import sys
import argparse
from indoor3d_util import DATA_PATH, collect_point_label
from pathlib import Path

parser = argparse.ArgumentParser('collect')
parser.add_argument('--meta_path', default='meta/',
                    help='Path to meta folder containing class and anno_paths')  # TODO could move this to PatrickData
parser.add_argument('--data_path', default=None,
                    help='If data path needs to change, set it here. Should point to data root')
args = parser.parse_args()

if args.data_path is not None:
    DATA_PATH = args.data_path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, f'{args.meta_path}anno_paths.txt'))]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

output_folder = os.path.join(ROOT_DIR, 'data/s3dis/stanford_indoor3d')
if args.meta_path != 'meta/':
    output_folder = os.path.join(ROOT_DIR, f'data/s3dis/{args.meta_path[5:-1]}')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        out_filename = elements[-3] + '_' + elements[-2] + '.npy'  # Area_1_hallway_1.npy
        if args.data_path is not None:
            collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy', update_args=args)
        else:
            collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except Exception as e:
        print(anno_path, 'ERROR!!')
