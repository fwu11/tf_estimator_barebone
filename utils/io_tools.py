##############################
# io_tools should include the functionality of the following:
# The Training dataset: PASCAL VOC 2012 + additional Berkeley segmentation data 
# tested on Restricted PASCAL VOC 2012 Validation dataset
# 1. read datasets
# 2. Do data processing such as normalization
# 3. Do categorical to the labels
# 4. data augmentation

import os
import math
import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter


class read_dataset(object):
    def __init__(self,img_dir,label_dir,file_path, batch_size, mode,base_size,crop_size, ignore_label,classes):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.file_path = file_path
        self.batch_size = batch_size
        self.mode = mode
        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.classes = classes
        #self.num_examples = None

        files = self.load_filenames(self.file_path,self.mode)
        self.x_filenames = []
        self.y_filenames = []
        for filename in files:
            self.x_filenames.append(os.path.join(self.img_dir, "{}.jpg".format(filename.strip())))
            self.y_filenames.append(os.path.join(self.label_dir, "{}.png".format(filename.strip())))
        
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
            dataset = dataset.map(lambda x_filenames,y_filenames: tuple(tf.py_func(self._val_transform,[x_filenames,y_filenames],[tf.uint8,tf.uint8])), num_parallel_calls=threads)
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

    def _standardize(self,img,mask):
        x = tf.to_float(img)
        # Normalize the image
        MEAN_IMAGE = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        STD_IMAGE = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        x = ((x / 255.0) - MEAN_IMAGE) / STD_IMAGE

        y = mask

        return x,y

    def _transform(self, x_filenames, y_filenames):
    
        img = Image.open(x_filenames).convert("RGB")
        mask = Image.open(y_filenames).convert("P")
        
        crop_size = self.crop_size
        base_size = self.base_size
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # random scale (short edge)
        short_size = random.randint(int(base_size*0.5), int(base_size*2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        
        # ??? maybe need the mask to be [height,width,channel]
        # final transform
        img = np.asarray(img)
        mask = np.array(mask).astype(np.uint8)
        #mask = np.squeeze(mask,axis=2)
        mask[np.where(mask == self.ignore_label)] = self.classes
        mask = mask[...,None]

        return img, mask

    def _val_transform(self, x_filenames, y_filenames):
        img = Image.open(x_filenames).convert("RGB")
        mask = Image.open(y_filenames).convert("P")
        
        crop_size = self.crop_size
        
        outsize = crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))

        # final transform
        img = np.asarray(img)
        mask = np.array(mask).astype(np.uint8)
        #mask = np.squeeze(mask,axis=2)
        mask[np.where(mask == self.ignore_label)] = self.classes
        mask = mask[...,None]

        return img, mask





