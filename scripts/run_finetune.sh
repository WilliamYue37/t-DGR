#!/bin/bash

python methods/finetune/train_finetune.py --dataset datasets/continual_world/ --ckpt_folder finetune_dry_run
python methods/finetune/test.py --models runs/finetune_dry_run/learner_ckpts/ --runs 100
