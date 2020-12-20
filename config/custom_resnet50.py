#!/usr/bin/env python

"""Module for training deeplabv3plus on camvid dataset."""

from glob import glob

import tensorflow as tf

IMG_SIZE = 576
BATCH_SIZE = 4
NUM_CLASSES = 4
LR = 0.0001
EPOCHS = 20
BACKBONE = 'resnet50'

DATASET_RESOLUCAO = f'{IMG_SIZE}x{IMG_SIZE}'
DATASET_PARTITION = '75_15_10'
PREPROC_EQ_HIST = 'hist-equal'
PREPROC_TRANSF_MORF = 'transf-morf'
PREPROC = PREPROC_EQ_HIST
DATASET_CONFIG = '_'.join([config for config in [DATASET_RESOLUCAO, DATASET_PARTITION, PREPROC] if config])
DATASET_CUSTOM_DIR = f'/content/drive/MyDrive/dataset/{DATASET_RESOLUCAO}/custom_{DATASET_CONFIG}'

# Sample Configuration
CONFIG = {
    # We mandate specifying project_name and experiment_name in every config
    # file. They are used for wandb runs if wandb api key is specified.
    'project_name': 'deeplabv3-plus',
    'experiment_name': 'custom-resnet',

    'train_dataset_config': {
        'images': sorted(glob(f'{DATASET_CUSTOM_DIR}/train/*')),
        'labels': sorted(glob(f'{DATASET_CUSTOM_DIR}/train_gt/*')),
        'height': IMG_SIZE, 'width': IMG_SIZE, 'batch_size': BATCH_SIZE
    },

    'val_dataset_config': {
        'images': sorted(glob(f'{DATASET_CUSTOM_DIR}/val/*')),
        'labels': sorted(glob(f'{DATASET_CUSTOM_DIR}/val_gt/*')),
        'height': IMG_SIZE, 'width': IMG_SIZE, 'batch_size': BATCH_SIZE
    },

    'strategy': tf.distribute.OneDeviceStrategy(device="/gpu:0"),
    'num_classes': NUM_CLASSES, 'backbone': BACKBONE, 'learning_rate': LR,

    'checkpoint_dir': "./checkpoints/",
    'checkpoint_file_prefix': "deeplabv3plus_with_resnet50_",

    'epochs': EPOCHS
}
