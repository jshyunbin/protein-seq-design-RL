import subprocess
from net.model import Model
from gym import Env, spaces 
from config import *
import warnings
from omegafold import pipeline
import omegafold.utils.protein_utils.residue_constants as rc
import random
import torch
from utils.data_utils import str2inputs
from omegafold import utils
import numpy as np 

warnings.filterwarnings('ignore')

def random_mutate(seq, N):
  pos = random.sample(range(len(seq)), N)
  mut = ''
  for i in range(len(seq)):
    if i in pos:
      mut = mut + random.choice(rc.restypes)
    else:
      mut = mut + seq[i]
  return mut

class SingleOpt(Env):
  def __init__(self, args):
    super().__init__()

    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(args.seq_len, 256))
    self.action_space = spaces.MultiDiscrete([args.seq_len, 20])
    self.name = args.name
    
    model_args, state_dict, forward_config = get_args(
      device=args.device,
      num_cycle=1,
      seq_len=args.seq_len,
    )

    model = Model(model_args, state_dict, forward_config)
    model.reward.load_state_dict(torch.load(args.saved_dir, map_location=args.device))
    model.eval()
    self.model = model.to(args.device)
    self.device = args.device

    self.ep = 0
    self.steps = 0
    self.state = None
    self.sequence = None
    self.wildtype = args.wildtype
    self.done_cond = args.done_cond
    self.stability_thr = args.stability_thr

    # logging
    self.reward = 0
    self.target = 0
    self.stability = 0
    self.n_mut = 0

  def reset(self):
    self.ep += 1
    self.steps = 0
    self.sequence = random_mutate(self.wildtype, random.choice(range(1, 5)))
    data = str2inputs(self.sequence, num_cycle=1)
    data = utils.recursive_to(data, device=self.device)
    with torch.no_grad():
      target, output = self.model(data)
    return output['node_repr'].detach().cpu().numpy()
  
  def step(self, action):
    self.steps += 1
    pos, aa = action
    self.sequence = self.sequence[:pos] + rc.restypes[aa] + self.sequence[pos+1:]
    data = str2inputs(self.sequence, num_cycle=1)
    data = utils.recursive_to(data, device=self.device)
    with torch.no_grad():
      target, output = self.model(data)
    self.state = output['node_repr'].detach().cpu().numpy()
    self.target = float(target.item())
    
    pipeline.save_pdb(
      pos14=output["final_atom_positions"],
      b_factors=output["confidence"] * 100,
      sequence=data[0]['p_msa'][0],
      mask=data[0]['p_msa_mask'][0],
      save_path=f"{self.name}.pdb",
      model=0
    )

    args = ('./FoldX', '--command=Stability', f'--pdb={self.name}.pdb')

    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    stats = popen.stdout.read().decode('utf-8')
    stats = stats.split("\n")
    stats = {o.split('=')[0].strip(): float(o.split('=')[1].strip()) for o in stats if '=' in o}
    self.stability = stats['Total']

    if self.stability < self.stability_thr:
      self.reward = target
    else:
      self.reward = -1
    
    self.n_mut = 0
    for i in range(len(self.sequence)):
      self.n_mut += 1 if self.sequence[i] != self.wildtype[i] else 0
    
    done = self.steps > self.done_cond['max_steps'] or target > self.done_cond['max_target'] or self.n_mut > self.done_cond['max_mutation']

    return self.state, self.reward, done, {'candidates': self.sequence, 'performance': self.target}