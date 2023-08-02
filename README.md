# t-DGR
A trajectory-based Deep Generative Replay method for lifelong learning in decision making tasks.

![Figure 1](images/Figure_1.svg)
![Figure 2](images/DGR.svg)

# Contents

- [Installation](#Installation)
- [Datasets](#Dataset)
- [Getting Started](#Getting-Started)
  - [Training](#Training)
  - [Evaluation](#Evaluation)
- [Citation](#Citation)


# Installtion
Please run the following commands in the given order to install the dependency for **LIBERO**.
```
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Then install the `libero` package:
```
pip install -e .
```

# Datasets
We provide high-quality human teleoperation demonstrations for the four task suites in **LIBERO**. To download the demonstration dataset, run:
```python
python benchmark_scripts/download_libero_datasets.py
```
By default, the dataset will be stored under the ```LIBERO``` folder and all four datasets will be downloaded. To download a specific dataset, use
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET
```
where ```DATASET``` is chosen from `[libero_spatial, libero_object, libero_100, libero_goal`.


# Getting Started

For a detailed walk-through, please either refer to the documentation or the notebook examples provided under the `notebooks` folder. In the following, we provide example scripts for retrieving a task, training and evaluation.

## Training
To start a lifelong learning experiment, please choose:
- `BENCHMARK` from `[LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_90, LIBERO_10]`
- `POLICY` from `[bc_rnn_policy, bc_transformer_policy, bc_vilt_policy]`
- `ALGO` from `[base, er, ewc, packnet, multitask]`

then run the following:

```shell
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python libero/lifelong/main.py seed=SEED \
                               benchmark_name=BENCHMARK \
                               policy=POLICY \
                               lifelong=ALGO
```
Please see the documentation for the details of reproducing the study results.

## Evaluation

By default the policies will be evaluated on the fly during training. If you have limited computing resource of GPUs, we offer an evaluation script for you to evaluate models separately.

```shell
python libero/lifelong/evaluate.py --benchmark BENCHMARK_NAME \
                                   --task_id TASK_ID \ 
                                   --algo ALGO_NAME \
                                   --policy POLICY_NAME \
                                   --seed SEED \
                                   --ep EPOCH \
                                   --load_task LOAD_TASK \
                                   --device_id CUDA_ID
```

# Citation
If you find **t-DGR** to be useful in your own research, please consider citing our paper: