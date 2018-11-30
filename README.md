# TF Estimator Barebone
TensorFlow project template with high-level API

## Examples

### CIFAR-10 with ResNet
Usage:
```bash
python trainer.py --dataset cifar10 --model cifar10_resnet --job-dir ./cifar10
python trainer.py --dataset cifar10 --model cifar10_resnet --mixup 1.0 --job-dir ./cifar10_mixup
```
Accurarcy: 94.09% without mixup, 94.89% with mixup

### PASCAL VOC augmented dataset with FCN, deeplab v2 and deeplab v3

deeplab v3:  
val mIOU = 71.36% after 120k steps  
batch = 4  
learning_rate=((63000, 80000, 100000), (0.0001, 0.00005, 0.00001, 0.000001))  
gpu = 1  
pretrain = Imagenet  
crop_size = 513  
base_size = 540  
output_stride = 16  
multi_grid = [1,2,4]

