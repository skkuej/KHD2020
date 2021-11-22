#!/bin/bash

## Fold 0 ##
python3 train.py --nf 0 -e 4 -l 0.0003 -t 0
python3 train.py --nf 0 -e 1 -l 0.00015 -f model0_epoch3
python3 train.py --nf 0 -e 1 -l 0.000075 -f model0_epoch4
python3 train.py --nf 0 -e 1 -l 0.000001 -f model0_epoch5
python3 train.py --nf 0 -e 1 -l 0.0000001 -f model0_epoch6 -a 1
python3 train.py --nf 0 -e 15 -l 0.00000001 -f model0_epoch7 -a 1
python3 train.py --nf 0 -e 9 -l 0.00000001 -f model0_epoch22 -a 2       # cutout


## Fold 3 ##
python3 train.py --nf 3 -b 8 -s 1 -m 'efficientnet-b5' -e 5 -l 0.0001 -t 1 --skip 1
python3 train.py --nf 3 -b 8 -a 1 -s 1 -m 'efficientnet-b5' -e 2 -l 0.00005 -f model1_epoch4
python3 train.py --nf 3 -b 8 -a 1 -s 1 -m 'efficientnet-b5' -e 7 -l 0.000001 -f model1_epoch6
python3 train.py --nf 3 -b 8 -a 5 -s 1 -m 'efficientnet-b5' -e 7 -l 0.0000005 -f model1_epoch13


## Fold 4 ##
python3 train.py --nf 4 -e 3 -l 0.0003 -t 2
python3 train.py --nf 4 -e 2 -l 0.00001 -f model2_epoch2
python3 train.py --nf 4 -e 2 -l 0.000001 -f model2_epoch4
python3 train.py --nf 4 -e 14 -l 0.0000001 -f model2_epoch6 -a 1
python3 train.py --nf 4 -e 20 -l 0.000001 -f model2_epoch20 -a 1
python3 train.py --nf 1 -e 8 -l 0.0000001 -f model2_epoch40 -a 3
