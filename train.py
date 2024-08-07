import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    # MRAugment specific arguments #################################
    parser.add_argument('--aug_on', default=True, action='store_true', help='This switch turns data augmentation on.')
    parser.add_argument('--aug_schedule', type=str, default='constant', help='Type of data augmentation strength scheduling. Options: constant, ramp, exp')
    parser.add_argument('--aug_delay', type=int, default=0, help='Number of epochs at the beginning of training without data augmentation.')
    parser.add_argument('--aug_strength', type=float, default=10.0, help='Augmentation strength')
    parser.add_argument('--aug_exp_decay', type=float, default=5.0, help='Exponential decay coefficient')
    parser.add_argument('--aug_interpolation_order', type=int, default=1, help='Order of interpolation filter used in data augmentation')
    parser.add_argument('--aug_upsample', default=False, action='store_true', help='Set to upsample before augmentation to avoid aliasing artifacts.')
    parser.add_argument('--aug_upsample_factor', type=int, default=2, help='Factor of upsampling before augmentation')
    parser.add_argument('--aug_upsample_order', type=int, default=1, help='Order of upsampling filter before augmentation')
    parser.add_argument('--aug_weight_translation', type=float, default=1.0, help='Weight of translation probability')
    parser.add_argument('--aug_weight_rotation', type=float, default=1.0, help='Weight of arbitrary rotation probability')
    parser.add_argument('--aug_weight_shearing', type=float, default=1.0, help='Weight of shearing probability')
    parser.add_argument('--aug_weight_scaling', type=float, default=1.0, help='Weight of scaling probability')
    parser.add_argument('--aug_weight_rot90', type=float, default=1.0, help='Weight of probability of rotation by multiples of 90 degrees')
    parser.add_argument('--aug_weight_fliph', type=float, default=1.0, help='Weight of horizontal flip probability')
    parser.add_argument('--aug_weight_flipv', type=float, default=1.0, help='Weight of vertical flip probability')
    parser.add_argument('--aug_max_translation_x', type=float, default=0.125, help='Maximum translation applied along the x axis as fraction of image width')
    parser.add_argument('--aug_max_translation_y', type=float, default=0.125, help='Maximum translation applied along the y axis as fraction of image height')
    parser.add_argument('--aug_max_rotation', type=float, default=180.0, help='Maximum rotation applied in either clockwise or counter-clockwise direction in degrees.')
    parser.add_argument('--aug_max_shearing_x', type=float, default=15.0, help='Maximum shearing applied in either positive or negative direction in degrees along x axis.')
    parser.add_argument('--aug_max_shearing_y', type=float, default=15.0, help='Maximum shearing applied in either positive or negative direction in degrees along y axis.')
    parser.add_argument('--aug_max_scaling', type=float, default=0.25, help='Maximum scaling applied as fraction of image dimensions.')
    parser.add_argument("--max_train_resolution", nargs="+", default=None, type=int, help="If given, training slices will be center cropped to this size if larger along any dimension.")

    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
