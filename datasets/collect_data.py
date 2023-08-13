import argparse
import numpy as np
import torch
import os

from metaworld.policies.sawyer_coffee_button_v2_policy import SawyerCoffeeButtonV2Policy
from metaworld.policies.sawyer_button_press_v2_policy import SawyerButtonPressV2Policy
from metaworld.policies.sawyer_drawer_open_v2_policy import SawyerDrawerOpenV2Policy
from metaworld.policies.sawyer_door_open_v2_policy import SawyerDoorOpenV2Policy
from metaworld.policies.sawyer_faucet_open_v2_policy import SawyerFaucetOpenV2Policy

from metaworld.policies.sawyer_hammer_v2_policy import SawyerHammerV2Policy
from metaworld.policies.sawyer_push_wall_v2_policy import SawyerPushWallV2Policy
from metaworld.policies.sawyer_faucet_close_v2_policy import SawyerFaucetCloseV2Policy
from metaworld.policies.sawyer_push_back_v2_policy import SawyerPushBackV2Policy
from metaworld.policies.sawyer_stick_pull_v2_policy import SawyerStickPullV2Policy
from metaworld.policies.sawyer_handle_press_side_v2_policy import SawyerHandlePressSideV2Policy
from metaworld.policies.sawyer_push_v2_policy import SawyerPushV2Policy
from metaworld.policies.sawyer_shelf_place_v2_policy import SawyerShelfPlaceV2Policy
from metaworld.policies.sawyer_window_close_v2_policy import SawyerWindowCloseV2Policy
from metaworld.policies.sawyer_peg_unplug_side_v2_policy import SawyerPegUnplugSideV2Policy

from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_rollouts', type=int, default=100, help='number of rollouts per task')
argparser.add_argument('--max_step', type=int, default=200, help='max step per rollout')
argparser.add_argument('--folder', type=str, default='expert_demo_data', help='folder to save the data')
args = argparser.parse_args()

env_names = ['hammer-v2', 'push-wall-v2', 'faucet-close-v2', 'push-back-v2', 'stick-pull-v2', 'handle-press-side-v2', 'push-v2', 'shelf-place-v2', 'window-close-v2', 'peg-unplug-side-v2']

policies = [SawyerHammerV2Policy(), SawyerPushWallV2Policy(), SawyerFaucetCloseV2Policy(), SawyerPushBackV2Policy(), SawyerStickPullV2Policy(), SawyerHandlePressSideV2Policy(), SawyerPushV2Policy(), SawyerShelfPlaceV2Policy(), SawyerWindowCloseV2Policy(), SawyerPegUnplugSideV2Policy()]

maxi = [0] * 10
errors = 0

lo, hi = None, None

def get_rollouts(task_id, env_name, policy, num_rollouts = 100, max_step = 200):
    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    i = 0
    while i < num_rollouts:
        rollout = []
        obs = env.reset()
        step = 0
        success = False
        while step < max_step - 1:
            a = policy.get_action(obs)
            a = a.astype(np.float32)
            # append one hot task id vector to obs
            obs = np.concatenate([obs, np.eye(10)[task_id]])
            global maxi
            maxi[task_id] = max(maxi[task_id], step)
            assert obs.shape == (49,)
            rollout.append([obs, a])
            obs, reward, done, info = env.step(a)

            step += 1
            if info['success']:
                assert len(rollout) >= 16
                obs = np.concatenate([obs, np.eye(10)[task_id]])
                while len(rollout) < max_step:
                    rollout.append([obs, a])
                success = True
                break

        if not success:
            global errors
            errors += 1
            continue
           
        path = f'{args.folder}/{env_name}'
        os.makedirs(path, exist_ok=True)
        torch.save(rollout, os.path.join(path, f'{i}.rollout'))
        i += 1

for i in range(len(env_names)):
    get_rollouts(i, env_names[i], policies[i], args.num_rollouts, args.max_step)

torch.save(maxi, f'{args.folder}/maxi.pt')

