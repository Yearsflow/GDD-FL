#!/bin/bash
python make_hdf5.py --dataset isic2020 --batch_size 64 --data_root /data14/tyc/dataset
python calculate_inception_moments.py --dataset isic2020 --data_root /data14/tyc/dataset