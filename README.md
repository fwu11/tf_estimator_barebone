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


Model | Validation mIOU | Steps | Batch | Learning rate | Output stide | Multi grid | L2 regularizer | Num GPU |
:------:|:------:|:-----:|:----:|:------------------------:|:--------:|:--------:|:--------:|:--------:|
deeplab v3 | 71.41% | 120k | 4 | (63k, 80k, 100k), (1e-4, 5e-5, 1e-5, 1e-6) | 16 | [1,2,4] | 0.0001 | 1

