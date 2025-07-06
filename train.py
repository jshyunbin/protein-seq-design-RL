import timeit
import torch 
import argparse
import torch.nn as nn
from net.model import Model
from utils.dataset import protein_data_builder
from torch.utils.data import DataLoader
from utils.loggings import *
from config import *
from omegafold.utils.protein_utils.residue_constants import *
import wandb
from omegafold.utils import recursive_to

parser = argparse.ArgumentParser()
parser.add_argument('--protein', type=int)
parser.add_argument('--num_cycle', type=int, default=1)
parser.add_argument('--device', type=str)
parser.add_argument('--save_name', type=str)
cmd_args = parser.parse_args()

data_names = ['IF1_ECOLI_Kelsic_2016', 'NUD15_HUMAN_Suiter_2020', 'REV_HV1H2_Fernandes_2016']
lengths = [72, 164, 116]
assert cmd_args.protein in range(len(data_names))

if not os.path.exists('saved/'):
  os.mkdir('saved')

total_epochs = 1
use_scheduler = True
log = True
device = cmd_args.device 

args, state_dict, forward_config = get_args(
  device=device,
  num_cycle=cmd_args.num_cycle,
  seq_len=lengths[cmd_args.protein],
)

model = Model(args, state_dict, forward_config)
model = model.to(device)

save_name = data_names[cmd_args.protein].split('_')[0] + '_' + cmd_args.save_name

train_dataset, test_dataset = protein_data_builder(data_names[cmd_args.protein], num_cycle=cmd_args.num_cycle)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

mse_loss = nn.MSELoss()
step_size = len(train_dataset) 

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-5)
if use_scheduler:
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=0.8)

if log: 
  wandb.init(project="Protein-optimization", entity="haewonc")
  wandb.watch(models=model)

def evaluate(state, loader):
  print()
  start_time = timeit.default_timer()
  total_loss = 0.0

  with torch.no_grad():
    for idx, data in enumerate(loader):
      input_data, target = data
      for d in input_data:
        for k in d:
          d[k] = d[k].squeeze(0)
      input_data = recursive_to(input_data, device=device)
      target = target.to(device).float()

      pred, _ = model(input_data)
      loss_val = mse_loss(pred.squeeze(0), target)

      loss_val = loss_val.item()
      total_loss += loss_val

      print_logs(state, epoch, total_epochs, idx, len(loader), timeit.default_timer()-start_time, loss_val, "MSE")

      start_time = timeit.default_timer()

    if log:
      wandb.log({f"{state}/mse": total_loss/len(loader)})

print(toYellow("FINETUNE STARTED\n"))

for epoch in range(total_epochs):
  print()
  model.train()
  start_time = timeit.default_timer()
  loss_vals = []
  for idx, data in enumerate(train_loader):
    optimizer.zero_grad()

    input_data, target = data
    for d in input_data:
      for k in d:
        d[k] = d[k].squeeze(0)
    input_data = recursive_to(input_data, device=device)
    target = target.to(device).float()

    pred, _ = model(input_data)
    loss_val = mse_loss(pred.squeeze(0), target)

    loss_val.backward()
    optimizer.step()
    if use_scheduler:
      scheduler.step()

    loss_val = loss_val.item()
    loss_vals.append(loss_val)

    print_logs('finetune', epoch, total_epochs, idx, len(train_loader), timeit.default_timer()-start_time, loss_val, "MSE")

    if log:
      wandb.log({"finetune/mse": loss_val})
    
    start_time = timeit.default_timer()

  model.eval()
  with torch.no_grad():
    evaluate("test", test_loader)
  
torch.save(model.reward.state_dict(), 'saved/{}.pt'.format(save_name))