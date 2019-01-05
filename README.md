# TF Estimator Barebone
TensorFlow project template with high-level API

## Examples
PASCAL VOC image segmentation  
DAVIS video segmentation
### PASCAL VOC augmented dataset with FCN, deeplab v2 and deeplab v3  
Usage:
```bash
python -u main.py --dataset cifar10 --model cifar10_resnet --job-dir ./cifar10

```
| Model | Validation mIOU | Batch | Learning rate | Output stide | Multi grid | L2 regularizer | Num GPU | Batchnorm | Nonlocal |
|:------:|:------:|:-----:|:----------------------:|:--------:|:--------:|:--------:|:--------:|:------:|:------:|
| deeplab v3 | 71.41% | 4 | (63k, 80k, 100k), (1e-4, 5e-5, 1e-5, 1e-6) | 16 | [1,2,4] | 0.0001 | 1 | frozen | 0 |
| deeplab v3 | 74.56% | 8 | (10k, 40k), (1e-3, 1e-4, 1e-5)  | 16 | [1,2,4] | 0.0001 | 1 | frozen | 3 |
| deeplab v3 | 75.04% | 8 | (10k, 40k), (1e-3, 1e-4, 1e-5) | 8 | [1,2,4] | 0.0001 | 1 | frozen | 3 |


### DAVIS with deeplab v3 + Nonlocal block

### Files
fcn_resnet101  
deeplab_v2 implements  
deeplab_v3  
deeplab_v3_2  
deeplab_v3_nonlocal  

