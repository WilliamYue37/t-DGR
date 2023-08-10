#!/bin/bash

python methods/packnet/train_packnet.py --dataset datasets/continual_world/ --ckpt_folder packnet_dry_run
python methods/packnet/test.py --models runs/packnet_dry_run/learner_ckpts/ --runs 100
