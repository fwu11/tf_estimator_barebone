# implement FCN in tensorflow/keras
# This implementation follows the FCN-resnet101
# validate on val set on PASCAL VOC 2012
# fwu11

import argparse
import tensorflow as tf
import os
import math
from utils.pascal_io_tools import read_dataset
from models.deeplab_v3 import model_fn,update_argparser
import warnings
warnings.filterwarnings('ignore')

def main(argv=None):
    update_argparser(parser)
    hparams = parser.parse_args(argv[1:])
    print(hparams)
    dataset_root = 'dataset/VOCdevkit/VOC2012'
    label_root = 'dataset'
    img_dir = os.path.join(dataset_root, "JPEGImages")
    label_dir = os.path.join(label_root,"SegmentationClassAug")
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
    )

    ws = None
    if hparams.warm_start:
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from="./models/resnet_v1_101/model.ckpt",
                                            vars_to_warm_start=['resnet_v1_101/(block)|(conv).*'])

    # build an estimator
    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir = hparams.job_dir,
        config = run_config,
        params= hparams,
        warm_start_from=ws)

    

    train_spec = tf.estimator.TrainSpec(
        input_fn = lambda: train_ds.get_input_fn(),
        max_steps = hparams.train_steps,
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
        '--job_dir', type=str, default='./models/20181129',
        help='Output directory for model and training stats.')
    parser.add_argument(
        '--train_steps', type=int, default=None,
        help='Training steps.')
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size to be used.')

    parser.add_argument(
        '--resnet_model', default="resnet_v1_101", 
        help="Resnet v1 101 model to use as feature extractor.")
    parser.add_argument(
        "--l2_regularizer", type=float, default=0.0001, 
        help="l2 regularizer parameter.")
    parser.add_argument(
        '--learning_rate', type=float, default=0.0001, 
        help="initial learning rate.")
    parser.add_argument(
        '--multi_grid', type=list, default=[1,2,4], 
        help="Spatial Pyramid Pooling rates")
    parser.add_argument(
        '--output_stride', type=int, default=16, 
        help="Spatial Pyramid Pooling rates")
        
    parser.add_argument(
        '--base_size', type=int, default=540,
        help='Base size to be used.')
    parser.add_argument(
        '--crop_size', type=int, default=513,
        help='Crop size after image preprocessing.')
    parser.add_argument(
        '--eval_base_size', type=int, default=540,
        help='Base size to be used.')
    parser.add_argument(
        '--eval_crop_size', type=int, default=513,
        help='Crop size after image preprocessing.')
    parser.add_argument(
        '--num_classes', type=int, default=21,
        help='Total number of classes in the dataset(including background,exclude void)')
    parser.add_argument(
        '--ignore_label', type=int, default=255,
        help='The void label')
    parser.add_argument(
        '--save_checkpoints_steps',
        help='Number of steps to save checkpoint',
        default=1000,
        type=int)
    parser.add_argument(
        '--random_seed',
        help='Random seed for TensorFlow',
        default=None,
        type=int)
    parser.add_argument(
        '--num_gpus',
        help='Number of GPUs for this task',
        default=4,
        type=int)
    parser.add_argument(
        '--warm_start',
        help='Load pretrained model',
        default= True,
        type = bool)

    tf.logging.set_verbosity("INFO")
    tf.app.run()