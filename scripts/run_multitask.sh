#!/bin/bash

python methods/multitask/train_multitask.py --dataset datasets/continual_world/ --ckpt_folder multitask_dry_run --epochs 1
python methods/multitask/test.py --model runs/multitask_dry_run/learner_ckpts/ --runs 1
