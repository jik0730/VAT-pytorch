#!/bin/sh

python train_semi.py --model_dir experiments/epsilon1
python train_semi.py --model_dir experiments/epsilon2
python train_semi.py --model_dir experiments/epsilon3
python train_semi.py --model_dir experiments/epsilon4
python train_semi.py --model_dir experiments/epsilon5