from net.model import Model
import random
import torch
from tqdm import tqdm
from omegafold.utils import recursive_to
from config import *
from net.envr import random_mutate
from utils.data_utils import str2inputs
from utils.dataset import ref_file
import csv

data_name = 'IF1_ECOLI_Kelsic_2016'
weight = 'saved/IF1_1.pt'
max_mutate = 3
iter = 3
dev = 'cuda:2'

def random_mutate_take(base_seq, max_n_mutate, top=100, device='cuda:1'):

    model = Model(*get_args(
        device=device,
        num_cycle=1,
        seq_len=len(base_seq),
    ))
    model.reward.load_state_dict(torch.load(weight, map_location=device))
    model.to(device=device)

    seq = []
    for _ in range(1000):
        n_mut = random.randint(1, max_n_mutate)
        seq.append(random_mutate(base_seq, n_mut))

    rewards = []
    for s in tqdm(seq):
        data = str2inputs(s, num_cycle=1)
        data = recursive_to(data, device=device)
        with torch.no_grad():
            reward, _ = model(data)
        reward = float(reward.item())
        rewards.append(reward)
    
    rewards, seq = zip(*sorted(zip(rewards, seq)))
    return seq[-top:], rewards[-top:]

def gen_alg(base_seq, iter, top=10, device='cuda:1'):
    model = Model(*get_args(
        device=device,
        num_cycle=1,
        seq_len=len(base_seq),
    ))
    model.reward.load_state_dict(torch.load(weight, map_location=device))
    model.to(device=device)

    seq = [base_seq for _ in range(top)]
    seq_list = seq.copy()
    rewards = None
    for i in range(iter):
        for s in seq_list:
            for _ in range(9):
                seq.append(random_mutate(s, 1))
        
        rewards = []
        print(f'iter: {i}')
        for s in tqdm(seq):
            data = str2inputs(s, num_cycle=1)
            data = recursive_to(data, device=device)
            with torch.no_grad():
                reward, _ = model(data)
            reward = float(reward.item())
            rewards.append(reward)
        
        rewards, seq = (list(t) for t in zip(*sorted(zip(rewards, seq))))
        seq = seq[-top:]
        seq_list = seq[-top:]
        rewards = rewards[-top:]
    return seq, rewards

def seq_distance(seq1, seq2):
    n_mut = 0
    for i in range(len(seq1)):
        n_mut += 1 if seq1[i] != seq2[i] else 0
    return n_mut

def top_10_csv(file_name):
    seq = []
    reward = []
    with open(file_name, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            seq.append(row[1])
            reward.append(float(row[2]))
    reward, seq = zip(*sorted(zip(reward, seq)))
    distances = []
    for i in range(len(seq)):
        for j in range(i+1, len(seq)):
            distances.append(seq_distance(seq[i], seq[j]))       
    avg = sum(reward[-10:]) / 10
    print(avg, sum(distances)/len(distances))

base_seq = ref_file['target_seq'][ref_file['DMS_id'].index(data_name)]
f = open(f"{data_name.split('_')[0]}_{iter}_gen_alg.csv", 'w')
res_seq, res_rew = gen_alg(base_seq, iter, device=dev)
for i in range(len(res_seq)):
    f.write(f"{i},{res_seq[i]},{res_rew[i]}\n")
f.close()