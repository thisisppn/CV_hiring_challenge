#!/bin/sh

wget https://s3-ap-southeast-1.amazonaws.com/he-public-data/dataset52bd6ce.zip -O dataset.zip
unzip dataset.zip

python train.py
