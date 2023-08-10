#!/bin/bash

python methods/t-dgr/train_DGR.py --dataset datasets/continual_world/ --ckpt_folder t-dgr_dry_run
python methods/t-dgr/test.py --models runs/t-dgr_dry_run/learner_ckpts/ --runs 100
