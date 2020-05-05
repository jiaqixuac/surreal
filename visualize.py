"""
This file try to load the checkpoint of the agent using the config file,
set up the agent as surreal/launch/launcher.py and render the process
Adapt from surreal/main/rollout.py

cd ~/Projects/RoboTurk/surreal
python visualize.py --folder ../surreal-subproc/BinCan32_4 --checkpoint learner.130000.pth
--env robosuite:SawyerPickPlaceCan/SawyerNutAssemblyRound --verbose --record

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
    default=None,
    help='which checkpoint want the agent to restore'
)
parser.add_argument(
    '--verbose',
    default=False,
    action='store_true',
    help='whether to print action range'
)
parser.add_argument(
    '--record',
    action='store_true'
)
parser.add_argument(
    '--record-every',
    type=int,
    default=1,
)

args = parser.parse_args()


def restore_model(folder, checkpoint=None):
    if checkpoint==None:
        assert os.path.exists(os.path.join(folder, 'checkpoint')), "No checkpoint folder found: {}".format(os.path.join(folder, 'checkpoint'))
        files = os.listdir(os.path.join(folder, 'checkpoint'))
        files.sort()
        for file in files:
            if file.endswith('ckpt') or file.endswith('pth'):
                checkpoint = file
        print("Given no checkpoints, use the checkpoint: {}".format(checkpoint))

    path_to_ckpt = os.path.join(folder, 'checkpoint',
                                checkpoint)  # suppose the checkpoint always comes along with its config file
    assert os.path.isfile(path_to_ckpt), "No checkpoint at: {}".format(path_to_ckpt)
    if path_to_ckpt.endswith('ckpt'):
        if not os.path.isfile(path_to_ckpt.replace('ckpt', 'pth')):
            import pickle
            with open(path_to_ckpt, 'rb') as f:
                data = pickle.load(f)
                torch.save({
                    'model': data['model']
                }, path_to_ckpt.replace('ckpt', 'pth'))
            print("Convert from ckpt file to pth file: {}".format(path_to_ckpt.replace('ckpt', 'pth')))
        path_to_ckpt = path_to_ckpt.replace('ckpt', 'pth')
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')
    print("Use device: {}, cuda available: {}".format(device, torch.cuda.is_available()))
    data = torch.load(path_to_ckpt, map_location=device)
    assert 'model' in data.keys()
    return data['model'], path_to_ckpt


if __name__ == "__main__":

    # ======== deprecated ========
    #     # config file may have been changed since recent modification
    #     # use code above to restore config

    #     learner_config = PPO_DEFAULT_LEARNER_CONFIG
    #     env_config = PPO_DEFAULT_ENV_CONFIG
    #     session_config = PPO_DEFAULT_SESSION_CONFIG
    # ======== deprecated end ========

    # restore configs
    configs = restore_config(os.path.join(args.folder, 'config.yml'))
    session_config, learner_config, env_config = \
        configs.session_config, configs.learner_config, configs.env_config

    if args.env and args.env != env_config.env_name:
        print("\n!!! Warning !!!")
        print("Change the environment from [{}] to [{}]\n".format(env_config.env_name, args.env))
        env_config.env_name = args.env
    else:
        print("\nThe environment is: [{}]\n".format(env_config.env_name))

    if args.record:
        record_folder = os.path.join(args.folder, 'video')
        os.makedirs(record_folder, exist_ok=True)
    else:
        record_folder = None

    # update the environment
    env_config.render = True
    env_config.sleep_time = 0.025
    env_config['control_freq'] = 100
    env_config['verbose'] = args.verbose

    env_config.video.record_video = args.record
    env_config.video.record_every = args.record_every
    env_config.video.save_folder = record_folder

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

    if args.folder:
        model, path_to_ckpt = restore_model(args.folder, args.checkpoint)
        agent.model.load_state_dict(model)
        print("\nLoaded checkpoint at: {}\n".format(path_to_ckpt))
    print("Agent created!")
    
    agent.main_eval()
