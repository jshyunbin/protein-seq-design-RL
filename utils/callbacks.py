from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from utils.evaluation import evaluate_policy
import numpy as np
import pandas as pd
import wandb
import heapq

class RewardLoggingCallback(BaseCallback):
  def __init__(self, save_dir, is_eval=False, verbose=0):
    super().__init__(verbose)
    self.cumul = 0
    self.save_dir = save_dir
    self.is_eval = is_eval
    self.sequences = []
  
  def _on_step(self) -> bool:
    # Log reward
    target = self.model.env.get_attr('target')
    sequence = self.model.env.get_attr('sequence')
    if self.is_eval:
      self.sequences.append([sequence[0], target[0]])
      if len(self.sequences) > 0 and len(self.sequences) % 100 == 0:
        pd.DataFrame(self.sequences).to_csv(f'{self.save_dir}/{len(self.sequences)//100}.csv')
    stability = self.model.env.get_attr('stability')
    reward = self.model.env.get_attr('reward')[0]
    self.cumul += reward
    n_mut = self.model.env.get_attr('n_mut')
    wandb.log({'Performance': target[0], 'Cumulative Reward': self.cumul, 'No. Mutation': n_mut[0], 'Stability': stability[0]})
    return True