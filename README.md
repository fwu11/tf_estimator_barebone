# Image and Video Segmentation with TF Estimators
TensorFlow Image and Video Segmentation project template with high-level API

## Examples
PASCAL VOC image segmentation  
DAVIS video segmentation
### PASCAL VOC augmented dataset with deeplab v3 and non-local blocks 
Usage:
```bash
python -u main.py --job-dir ./models/deeplabv3_nonlocal

```
| Model | Validation mIOU | Batch | Learning rate | Output stride | Multi grid | L2 regularizer | Num GPU | Batchnorm | Nonlocal |
|:------:|:------:|:-----:|:----------------------:|:--------:|:--------:|:--------:|:--------:|:------:|:------:|
| deeplab v3 | 71.41% | 4 | (63k, 80k, 100k), (1e-4, 5e-5, 1e-5, 1e-6) | 16 | [1,2,4] | 0.0001 | 1 | frozen | 0 |
| deeplab v3 | 74.56% | 8 | (10k, 40k), (1e-3, 1e-4, 1e-5)  | 16 | [1,2,4] | 0.0001 | 1 | frozen | 3 |
| deeplab v3 | 75.04% | 8 | (10k, 40k), (1e-3, 1e-4, 1e-5) | 8 | [1,2,4] | 0.0001 | 1 | frozen | 3 |


### DAVIS 2016 with deeplab v3 + Nonlocal block


