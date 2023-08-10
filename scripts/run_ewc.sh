#!/bin/bash

python methods/ewc/train_ewc.py --dataset datasets/continual_world/ --ckpt_folder ewc_dry_run
python methods/ewc/test.py --models runs/ewc_dry_run/learner_ckpts/ --runs 100
