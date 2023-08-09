#!/bin/bash

python methods/t-dgr/train_DGR.py --dataset datasets/continual_world/ --ckpt_folder dry_run --benchmark cw10 --warmup 1 --timestep 1 --steps 1 --epochs 1
python methods/t-dgr/test.py --models runs/dry_run/learner_ckpts/ --runs 1 --benchmark cw10
