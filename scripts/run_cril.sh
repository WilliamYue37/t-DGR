#!/bin/bash

python methods/cril/train_CRIL.py --dataset datasets/continual_world/ --ckpt_folder cril_dry_run
python methods/cril/test.py --models runs/cril_dry_run/learner_ckpts/ --runs 100
