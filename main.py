# implement FCN in tensorflow/keras
# This implementation follows the FCN-resnet101
# Train on the 8,498 images of SBD train
# validate on the non-interecting set on PASCAL VOC 2012
# test on Pascal VOC 2012
# fwu11

import argparse
import tensorflow as tf
import os
import numpy as np
import math
from utils.io_tools import read_dataset
from models.fcn_resnet101 import model_fn
from utils.loss import *


def main(argv=None):
    '''
    REMOTE_URL = 'http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    LOCAL_DIR = os.path.join('data/cifar10/')
    ARCHIVE_NAME = 'VOCtrainval_11-May-2012.tar'
    DATA_DIR = 'cifar-10-batches-py/'
    '''
    '''
    #get PASCAL VOC 2012 data for the first time
    wget http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar -xf VOCtrainval_11-May-2012.tar
    mv VOCdevkit dataset/
    print('finish getting data')
    '''

    hparams = parser.parse_args(argv[1:])   
    dataset_root = 'dataset/VOCdevkit/VOC2012'
    img_dir = os.path.join(dataset_root, "JPEGImages")
    label_dir = os.path.join(dataset_root,"SegmentationClass")
    train_file_path = os.path.join(dataset_root,"ImageSets/Segmentation/train.txt")
    val_file_path = os.path.join(dataset_root,"ImageSets/Segmentation/val.txt")


    # create training dataset object
    train_ds = read_dataset(img_dir,
                            label_dir,
                            train_file_path, 
                            hparams.train_batch_size, 
                            "train",
                            hparams.base_size,
                            hparams.crop_size, 
                            hparams.ignore_label,
                            hparams.num_classes)

    # evaluation data
    eval_ds = read_dataset(img_dir,
                            label_dir,
                            val_file_path, 
                            hparams.eval_batch_size, 
                            "eval",
                            hparams.eval_base_size,
                            hparams.eval_crop_size, 
                            hparams.ignore_label,
                            hparams.num_classes)
    

    #session_config = tf.ConfigProto()
    '''
    session_config.gpu_options.allow_growth = args.allow_growth
    if args.xla:
        session_config.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1)
    '''

    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus = hparams.num_gpus)
    run_config = tf.estimator.RunConfig(
        model_dir=hparams.job_dir,
        tf_random_seed=hparams.random_seed,
        save_checkpoints_steps=hparams.save_checkpoints_steps,
        train_distribute = strategy,
        #session_config=session_config,
    )   

    num_train_examples = train_ds.num_examples

    train_steps_per_epoch = math.ceil(num_train_examples / hparams.train_batch_size)
    # build an estimator
    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir = hparams.job_dir,
        config = run_config,
        params= {
          'num_training_examples': num_train_examples,
          'train_epoch': hparams.train_epoch,
          'crop_size': hparams.crop_size,
          'num_classes':hparams.num_classes,
        })

    

    train_spec = tf.estimator.TrainSpec(
      input_fn = lambda: train_ds.get_input_fn(),
      max_steps = train_steps_per_epoch * hparams.train_epoch,
    )

    num_eval_examples = eval_ds.num_examples
    eval_steps_per_epoch = math.ceil(num_eval_examples / hparams.eval_batch_size)

    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: eval_ds.get_input_fn(),
        steps = eval_steps_per_epoch,
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    # Setup input args parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir', type=str, default='./models/4gpus',
        help='Output directory for model and training stats.')
    parser.add_argument(
        '--train-epoch', type=int, default=100,
        help='Training epoch.')
    parser.add_argument(
        '--train-batch-size', type=int, default=8,
        help='Batch size to be used.')
    parser.add_argument(
        '--eval-batch-size', type=int, default=8,
        help='Batch size to be used.')
    parser.add_argument(
        '--base-size', type=int, default=280,
        help='Base size to be used.')
    parser.add_argument(
        '--crop-size', type=int, default=240,
        help='Crop size after image preprocessing.')
    parser.add_argument(
        '--eval-base-size', type=int, default=280,
        help='Base size to be used.')
    parser.add_argument(
        '--eval-crop-size', type=int, default=240,
        help='Crop size after image preprocessing.')
    parser.add_argument(
        '--num-classes', type=int, default=21,
        help='Total number of classes in the dataset(including background,exclude void)')
    parser.add_argument(
        '--ignore-label', type=int, default=255,
        help='The void label')
    parser.add_argument(
        '--save-checkpoints-steps',
        help='Number of steps to save checkpoint',
        default=1000,
        type=int)
    parser.add_argument(
        '--random-seed',
        help='Random seed for TensorFlow',
        default=None,
        type=int)
    parser.add_argument(
        '--num-gpus',
        help='Number of GPUs for this task',
        default=4,
        type=int)
    # Performance tuning parameters
    parser.add_argument(
        '--allow-growth',
        help='Whether to enable allow_growth in GPU_Options',
        default=False,
        type=bool)
    parser.add_argument(
        '--xla',
        help='Whether to enable XLA auto-jit compilation',
        default=False,
        type=bool)

    tf.app.run()