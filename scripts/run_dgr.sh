#!/bin/bash

python methods/dgr/train_DGR.py --dataset datasets/continual_world/ --ckpt_folder dgr_dry_run
python methods/dgr/test.py --models runs/dgr_dry_run/learner_ckpts/ --runs 100
