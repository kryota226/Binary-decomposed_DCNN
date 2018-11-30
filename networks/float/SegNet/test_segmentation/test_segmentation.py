# -*- coding: utf-8 -*-
'''
test_segmentation.py
==================================================
Evaluation tool for semantic segmentation task.
'''
from __future__ import print_function
import argparse
import imp
import os
import glob

import cv2
import numpy as np

import seg_evaluater


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--testset', required=True,
        help='Pairs of image and label sample definition text file.'
    )
    parser.add_argument(
        '--labelset', required=True,
        help='Pairs of image and label sample definition text file.'
    )
    parser.add_argument(
        '--palette', required=True,
        help='Color to paint in a class map.'
    )
    parser.add_argument(
        '--save_dir', required=True,
        help='Color to paint in a class map.'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Color to paint in a class map.'
    )
    return parser.parse_args()

def load_module(module_path):
    if not os.path.isfile(module_path):
        raise IOError('Not found a palette: {}'.format(module_path))
    head, tail = os.path.split(module_path)
    module_name = os.path.splitext(tail)[0]
    info = imp.find_module(module_name, [head])
    module = imp.load_module(module_name, *info)
    return np.asarray(module.palette, dtype='uint8')

def visualize(class_map, palette, save_file):
    assert class_map.ndim == 2
    map_height, map_width = class_map.shape
    canvas = np.ndarray((map_height, map_width, 3), dtype='uint8')
    for label in range(len(palette)):
        canvas[class_map == label] = palette[label]
    cv2.imwrite(save_file, canvas)

def main():
    args = parse_arguments()
    palette = load_module(args.palette)
    image_files = sorted(glob.glob(args.testset))
    label_files = sorted(glob.glob(args.labelset))

    assert 0 < len(image_files), len(image_file)
    assert 0 < len(label_files), len(label_files)
    if args.visualize and not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    accuracies = np.array((0.0, 0.0, 0.0), dtype='float')
    for x_file, y_file in zip(image_files, label_files):
        class_map = np.load(x_file)
        label_map = cv2.imread(y_file, cv2.IMREAD_GRAYSCALE)
        label_map = cv2.resize(label_map, (class_map.shape[1], class_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        accuracies += seg_evaluater.calc_accuracies(class_map, label_map, n_classes=13)
        if args.visualize:
            filename = os.path.splitext(os.path.basename(x_file))[0]
            save_file = os.path.join(args.save_dir, '{}.png'.format(filename))
            visualize(class_map, palette, save_file)
    print('Global accuracy = {:0.4f} [%]'.format(100. * accuracies[0] / len(image_files)))
    print('Class accuracy = {:0.4f} [%]'.format(100. * accuracies[1] / len(image_files)))
    print('Mean IoU = {:0.4f} [%]'.format(100. * accuracies[2] / len(image_files)))

if __name__ == '__main__':
    main()