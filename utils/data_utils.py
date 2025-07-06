from typing import Tuple, List
import torch
from omegafold.pipeline import *

def str2inputs(
        seq: str,
        num_pseudo_msa: int = 5,
        mask_rate: float = 0.12,
        num_cycle: int = 10,
        deterministic: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:

    seq = seq.replace("Z", "E").replace("B", "D").replace("U", "C")
    aatype = torch.LongTensor(
        [rc.restypes_with_x.index(aa) if aa != '-' else 21 for aa in seq]
    )
    mask = torch.ones_like(aatype).float()
    assert torch.all(aatype.ge(0)) and torch.all(aatype.le(21)), \
        f"Only take 0-20 amino acids as inputs with unknown amino acid " \
        f"indexed as 20"

    num_res = len(aatype)
    data = list()
    g = None
    if deterministic:
        g = torch.Generator()
        g.manual_seed(num_res)
    for _ in range(num_cycle):
        p_msa = aatype[None, :].repeat(num_pseudo_msa, 1)
        p_msa_mask = torch.rand(
            [num_pseudo_msa, num_res], generator=g
        ).gt(mask_rate)
        p_msa_mask = torch.cat((mask[None, :], p_msa_mask), dim=0)
        p_msa = torch.cat((aatype[None, :], p_msa), dim=0)
        p_msa[~p_msa_mask.bool()] = 21
        data.append({"p_msa": p_msa, "p_msa_mask": p_msa_mask})
    
    return data
