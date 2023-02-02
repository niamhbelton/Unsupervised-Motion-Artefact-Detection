
# IGD

*Not the official repo*

## Dataset

[**Download the MVTec AD dataset**](https://www.mvtec.com/company/research/datasets/mvtec-ad)



parser.add_argument('-n', '--num', nargs='+', type=int, help='<Required> Set flag', required=True)
parser.add_argument('-sr', '--sample_rate', default=1, type=float)
parser.add_argument( '--dataset', default='mnist', type=str)
parser.add_argument( '--exp_name',type=str)


- num is the normal class, all other classes are anomalies. Note: the first class is '1' (not '0')
- sr sample rate
- dataset, choices=['mnist', 'cifar10', 'mvtec']
- exp_name is the experiment name


Note:
change the train sizes in  p256.m_ssim_main

change the device

tests on the test set every 20 epochs
  
early stopping with patience of 5

iterates over five seeds

reduced batch size due to memory requirements

images for cifar and mnist are upsampled to 256 x 256 so that no changes are made to the architecture

Example command to run for cifar10 dataset
```
python -u -m p256.m_ssim_main --num 1 2 3 4 5 6 7 8 9 10 --dataset cifar10 --exp_name testing >> output1
```
