"""
This file try to load the checkpoint of the agent using the config file
set up the agent as surreal/launch/launcher.py
and render the process

first convert ckpt to pth using pickle.load and torch.save
cd ~/Projects/RoboTurk/surreal
#CUDA_VISIBLE_DEVICES="" python visualize.py --env robosuite:SawyerPickPlaceCan --checkpoint ../surreal-subproc/BinPicking32_1/checkpoint/learner.55000.pth#
#CUDA_VISIBLE_DEVICES="" python visualize.py --folder ../surreal-subproc/BinPicking32_4 --checkpoint ../surreal-subproc/BinPicking32_4/checkpoint/learner.130000.pth
#python visualize.py --folder ../surreal-subproc/BinPicking32_4 --checkpoint ../surreal-subproc/BinPicking32_4/checkpoint/learner.130000.pth --env robosuite:SawyerPickPlaceCan --verbose
python visualize.py --folder ../surreal-subproc/BinCan32_4 --checkpoint learner.130000.pth --env robosuite:SawyerPickPlaceCan --verbose

# playback in robosuite
cd ~/Projects/RoboTurk/robosuite/robosuite/scripts
python playback_demonstrations_from_hdf5.py --folder /home/jqxu/data/RoboTurk/RoboTurkPilot/bins-Can/
"""
import argparse
import os
import torch

from surreal.agent import PPOAgent
from surreal.main.ppo_configs_roboturk import (
    PPO_DEFAULT_LEARNER_CONFIG,
    PPO_DEFAULT_ENV_CONFIG,
    PPO_DEFAULT_SESSION_CONFIG
)
from surreal.env import make_env, make_env_config

from benedict import BeneDict


def restore_config(path_to_config):
    """
    Loads a config from a file.
    """
    configs = BeneDict.load_yaml_file(path_to_config)
    return configs


parser = argparse.ArgumentParser()
parser.add_argument(
    '--env', 
    type=str, 
    default=None,
    help='name of the environment'
)
parser.add_argument(
    "--folder", 
    type=str, 
    required=True,
    help='the folder contain config info'
)
parser.add_argument(
    '--checkpoint',
    type=str,
    required=True,
    help='which checkpoint want the agent to restore'
)
parser.add_argument(
    '--verbose',
    default=False,
    action='store_true',
    help='whether to print action range'
)

args = parser.parse_args()


if __name__ == "__main__":
    
    # restore configs
    configs = restore_config(os.path.join(args.folder, 'config.yml'))
    session_config, learner_config, env_config = \
        configs.session_config, configs.learner_config, configs.env_config

# ======== deprecated ========
#     # config file may have been changed since recent modification
#     # use code above to restore config

#     learner_config = PPO_DEFAULT_LEARNER_CONFIG
#     env_config = PPO_DEFAULT_ENV_CONFIG
#     session_config = PPO_DEFAULT_SESSION_CONFIG 
# ======== deprecated end ========
    
#     print("The environment is: [{}]".format(args.env))
#     env_config.env_name = args.env

    if args.env and args.env != env_config.env_name:
        print("\n!!! Warning !!!")
        print("Change the environment from [{}] to [{}]\n".format(env_config.env_name, args.env))
        env_config.env_name = args.env
    else:
        print("\nThe environment is: [{}]\n".format(env_config.env_name))

    env_config.render = True
    env_config.sleep_time = 0.025
    env_config['control_freq'] = 100
    env_config['verbose'] = args.verbose
    env_config = make_env_config(env_config, 'eval')
    
    agent_mode = 'eval_deterministic_local'
    agent = PPOAgent(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config,
        agent_id=0,
        agent_mode=agent_mode,
        render=True
    )

    if args.checkpoint:
        checkpoint = os.path.join(args.folder, 'checkpoint', args.checkpoint) # suppose the checkpoint always comes along with its config file
        assert os.path.isfile(checkpoint), "No checkpoint at: {}".format(checkpoint)
        if checkpoint.endswith('ckpt'):
            if not os.path.isfile(checkpoint.replace('ckpt', 'pth')):
                import pickle
                with open(checkpoint, 'rb') as f:
                    data = pickle.load(f)
                    torch.save({
                        'model': data['model']
                    }, checkpoint.replace('ckpt', 'pth'))
                print("Convert from ckpt file to pth file: {}".format(checkpoint.replace('ckpt', 'pth')))
            checkpoint = checkpoint.replace('ckpt', 'pth')
        device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')
        print("Use device: {}, cuda available: {}".format(device, torch.cuda.is_available()))
        data = torch.load(checkpoint, map_location=device)
        assert 'model' in data.keys()
        agent.model.load_state_dict(data['model'])
        print("\nLoaded checkpoint at: {}\n".format(checkpoint))
    print("Agent created!")
    
    agent.main_eval()
