# implement FCN in tensorflow/keras
# This implementation follows the FCN-resnet101
# 
# validate on the non-interecting set on PASCAL VOC 2012
# test on Pascal VOC 2012
# fwu11

import argparse
import tensorflow as tf
import os
import numpy as np
import math
from utils.io_tools import read_dataset
from models.deeplab_resnet101 import model_fn
from utils.loss import *
import warnings
warnings.filterwarnings('ignore')

def main(argv=None):

    hparams = parser.parse_args(argv[1:])   
    dataset_root = 'dataset/VOCdevkit/VOC2012'
    label_root = 'dataset'
    img_dir = os.path.join(dataset_root, "JPEGImages")
    label_dir = os.path.join(label_root,"SegmentationClassAug")
    #train_file_path = os.path.join(dataset_root,"ImageSets/Segmentation/train.txt")
    train_file_path = os.path.join(label_root,"trainaug.txt")
    val_file_path = os.path.join(dataset_root,"ImageSets/Segmentation/val.txt")


    # create training dataset object
    train_ds = read_dataset(img_dir,
                            label_dir,
                            train_file_path, 
                            hparams.batch_size, 
                            "train",
                            hparams.base_size,
                            hparams.crop_size, 
                            hparams.ignore_label,
                            hparams.num_classes)

    # evaluation data
    eval_ds = read_dataset(img_dir,
                            label_dir,
                            val_file_path, 
                            hparams.batch_size, 
                            "eval",
                            hparams.eval_base_size,
                            hparams.eval_crop_size, 
                            hparams.ignore_label,
                            hparams.num_classes)
    

    if hparams.num_gpus == 0:
        strategy = tf.contrib.distribute.OneDeviceStrategy('device:CPU:0')
    elif hparams.num_gpus == 1:
        strategy = tf.contrib.distribute.OneDeviceStrategy('device:GPU:0')
    else:
        strategy = tf.contrib.distribute.MirroredStrategy(num_gpus = hparams.num_gpus)
    
    run_config = tf.estimator.RunConfig(
        model_dir=hparams.job_dir,
        tf_random_seed=hparams.random_seed,
        save_checkpoints_steps=hparams.save_checkpoints_steps,
        train_distribute = strategy,
        #session_config=session_config,
    )   

    num_train_examples = train_ds.num_examples

    train_steps_per_epoch = math.ceil(num_train_examples / hparams.batch_size)
    
    ws = None
    if hparams.warm_start:
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from="./models/resnet_v1_101.ckpt",
                                            vars_to_warm_start="resnet.*")

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
        },
        warm_start_from=ws)

    

    train_spec = tf.estimator.TrainSpec(
      input_fn = lambda: train_ds.get_input_fn(),
      max_steps = train_steps_per_epoch * hparams.train_epoch,
    )

    num_eval_examples = eval_ds.num_examples
    eval_steps_per_epoch = math.ceil(num_eval_examples / hparams.batch_size)

    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: eval_ds.get_input_fn(),
        steps = eval_steps_per_epoch,
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    # Setup input args parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir', type=str, default='./models/new',
        help='Output directory for model and training stats.')
    parser.add_argument(
        '--train-epoch', type=int, default=150,
        help='Training epoch.')
    parser.add_argument(
        '--batch-size', type=int, default=8,
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
        default=1,
        type=int)
    # load pretrained model or not
    parser.add_argument(
        '--warm-start',
        help='Load pretrained model',
        default= True,
        type = bool)

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

    tf.logging.set_verbosity("INFO")
    tf.app.run()