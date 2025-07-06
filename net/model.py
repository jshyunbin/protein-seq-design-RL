import torch 
import torch.nn as nn 
import torch.nn.functional as F
import omegafold as of
import omegafold.pipeline as pipeline
from omegafold import utils
from omegafold.utils import residue_constants as rc
from config import get_args
from utils.data_utils import str2inputs
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

def create_pyg_data(node_reprs, coords, M):
  N = coords.shape[0]
  dist_matrix = torch.cdist(coords, coords)
  _, nearest_indices = torch.topk(dist_matrix, M + 1, largest=False, sorted=True)
  nearest_indices = nearest_indices[:, 1:]
  row = torch.arange(N).view(-1, 1).repeat(1, M).view(-1).to(node_reprs.device)
  col = nearest_indices.reshape(-1)
  edge_index = torch.stack([row, col])
  edge_attr = dist_matrix[row, col].unsqueeze(-1) 
  data = Data(x=node_reprs, edge_index=edge_index, edge_attr=edge_attr)

  return data

class RewardGNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = GCNConv(256, 128)
    self.conv2 = GCNConv(128, 64)
    self.fc = torch.nn.Linear(64, 1)

  def forward(self, data):
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
    x = self.conv1(x, edge_index, edge_weight=edge_weight)
    x = torch.relu(x)
    x = self.conv2(x, edge_index, edge_weight=edge_weight)
    x = torch.relu(x)
    x = global_mean_pool(x, batch=data.batch)
    x = self.fc(x)
    return x


class Model(nn.Module):

  def __init__(self, args, state_dict, forward_config):
    super().__init__()
    self.args = args
    self.cfg = of.make_config(args.model)
    model = of.OmegaFold(self.cfg)
    model.load_state_dict(state_dict)
    for k in model.parameters():
      k.requires_grad = False
    self.model = model.eval()
    self.reward = RewardGNN()
    self.fwd_cfg = forward_config
    self.device = args.device
  
  def forward(self, data):
    with torch.no_grad():
      output = self.model(
        inputs=data,
        fwd_cfg=self.fwd_cfg
      )
      atom_mask = rc.restype2atom_mask.to(self.device)[data[0]['p_msa'][..., 0, :]]
      coords = utils.create_pseudo_beta(output["final_atom_positions"], atom_mask)
      data = create_pyg_data(output['node_repr'], coords, self.args.M)
    reward = self.reward(data)
    return reward, output
    

def test_model():

  args, state_dict, forward_config = get_args(
    device='cuda:0',
    num_cycle=1,
    seq_len=101
  )

  seq = 'MTASAQPRGRRPGVGVGVVVTSCKHPRCVLLGKRKRSVGAGSFQLPGGHLEFGETWEECAQRETWEEAALHLKNVHFASVVNSFIEKENYHYVTILMKGEVDVTHDSEPKNVEPEKNESWEWVPWEELPPLDQLFWGLRCLKEQGYDPFKEDLNHLVGYKGNHL'

  data = str2inputs(seq, num_cycle=1)
  data = utils.recursive_to(data, device='cuda:0')
  model = Model(args, state_dict, forward_config)
  model = model.to('cuda:0')
  model(data)