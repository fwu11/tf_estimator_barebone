#########################################
# This is a DAVIS 2017 data loader
# 1. load data
# 2. resize
# 3. flip and gamma augmentation

import os
import math
import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter

class read_dataset(object):
    def __init__(self,img_dir,label_dir,file_path, batch_size, mode):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.file_path = file_path
        self.batch_size = batch_size
        self.mode = mode

        files = self.load_filenames(self.file_path,self.mode)
        self.x_filenames = []
        self.y_filenames = []
        for f in files:
            x_foldername = os.path.join(img_dir, f.strip())
            y_foldername = os.path.join(label_dir, f.strip())
            x_list = os.listdir(x_foldername)
            y_list = os.listdir(y_foldername)
            for x in x_list:
                self.x_filenames.append(os.path.join(x_foldername, x))
            for y in y_list:
                self.y_filenames.append(os.path.join(y_foldername, y))
        
        self.num_examples = len(self.x_filenames)

    def get_input_fn(self,threads=5):
        
        # Create a dataset from the filenames and labels
        # (filename, label)
        dataset = tf.data.Dataset.from_tensor_slices((self.x_filenames, self.y_filenames))

        # Map our preprocessing function to every element in our dataset, taking advantage of multithreading
        # (image_resized, label)
        if self.mode == "train":
            dataset = dataset.map(lambda x_filenames,y_filenames: tuple(tf.py_func(self._transform,[x_filenames,y_filenames],[tf.uint8,tf.uint8])), num_parallel_calls=threads)
            dataset = dataset.map(self._standardize, num_parallel_calls=threads)
            dataset = dataset.shuffle(self.num_examples)
        else:
            dataset = dataset.map(lambda x_filenames,y_filenames: tuple(tf.py_func(self._transform,[x_filenames,y_filenames],[tf.uint8,tf.uint8])), num_parallel_calls=threads)
            dataset = dataset.map(self._standardize, num_parallel_calls=threads)

        # It's necessary to repeat our data for all epochs
        # (image_resized_batch, label_batch)
        dataset = dataset.repeat().batch(self.batch_size).prefetch(1)
        return dataset

    def load_filenames(self,data_txt_file,mode):
        print("Read " + mode + " filenames")
        f = open(data_txt_file,'r')
        lines = f.readlines()
        f.close()
        return lines

    # normalize and gamma correction
    def _standardize(self,img,mask):
        x = tf.to_float(img)

        MEAN_IMAGE = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        STD_IMAGE = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        x = x/255.0
        factor = tf.random_uniform(shape=[], minval=-0.1, maxval=0.1, dtype=tf.float32)
        gamma = tf.log(0.5 + 1 / math.sqrt(2) * factor) / tf.log(0.5 - 1 / math.sqrt(2) * factor)
        # Perform the gamma correction
        x = x ** gamma

        # Normalize the image
        x = (x - MEAN_IMAGE) / STD_IMAGE
        y = mask

        return x,y

    # Data Augmentation train and val
    # 1.random_resize (batch size of 1)
    # 2.flip
    def _transform(self, x_filenames, y_filenames):
    
        img = Image.open(x_filenames).convert("RGB")
        mask = Image.open(y_filenames).convert("P")
        
        crop_size = self.crop_size
        base_size = self.base_size
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # random resize
        scale_factor = np.random.uniform(low = 0.7, high = 1.3)
        w, h = img.size
        ow = int(1.0*w*scale_factor)
        oh = int(1.0*h*scale_factor)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # mask to be [height,width,channel]
        # final transform
        img = np.asarray(img)
        mask = np.array(mask).astype(np.uint8)
        mask = mask[...,None]

        return img, mask