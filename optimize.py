import os
import time
import json
import shutil
import wandb
import argparse
from stable_baselines3 import SAC, PPO
from config import get_optimize_args
from utils.loggings import *
from net.envr import SingleOpt
from wandb.integration.sb3 import WandbCallback
from utils.callbacks import RewardLoggingCallback, WandbEvalCallback
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--protein', type=int)
parser.add_argument('--device', type=str)
parser.add_argument('--thr', type=int)

cmd_args = parser.parse_args()

data_names = ['IF1_ECOLI_Kelsic_2016', 'NUD15_HUMAN_Suiter_2020', 'REV_HV1H2_Fernandes_2016']
lengths = [72, 164, 116]
weights = ['saved/IF1_1.pt', 'saved/NUD15_1.pt', 'saved/REV_1.pt']
assert cmd_args.protein in range(len(data_names))
device = cmd_args.device
name = data_names[cmd_args.protein].split('_')[0]
save_dir = '{}_{}'.format(name, time.strftime('%d_%H_%M', time.localtime(time.time())))

if not os.path.exists('policy'):
  os.mkdir('policy')
if not os.path.exists('results'):
  os.mkdir('results')
os.mkdir('policy/{}'.format(save_dir))
os.mkdir('results/{}'.format(save_dir))

ref_file = load_dataset('ICML2022/ProteinGym', data_files='ProteinGym_reference_file_substitutions.csv', split='train')
wildtype = ref_file['target_seq'][ref_file['DMS_id'].index(data_names[cmd_args.protein])]

args = get_optimize_args(device, name, wildtype, lengths[cmd_args.protein], weights[cmd_args.protein], cmd_args.thr)
run = wandb.init(project="Protein-optimization", entity="haewonc")

env = SingleOpt(args)

model = PPO("MlpPolicy", env, verbose=1, device=device) # verbose 0 to not show
model.learn(total_timesteps=args.total_steps, callback=[WandbCallback(model_save_path='policy/'+save_dir), RewardLoggingCallback('results/{}'.format(save_dir))])