import argparse
import os
from omegafold.pipeline import _load_weights

def get_args(device, num_cycle, seq_len, embed_dim=256, nearest_M=10):
  args = argparse.Namespace(
    num_cycle=num_cycle,
    subbatch_size=None,
    device=device,
    model=2,
    weights_file=None,
    weights="https://helixon.s3.amazonaws.com/release1.pt",
    pseudo_msa_mask_rate=0.12,
    num_pseudo_msa=15,
    allow_tf32=True,
    seq_len=seq_len,
    embed_dim=embed_dim,
    M=nearest_M
  )

  if args.model == 1:
    weights_url = "https://helixon.s3.amazonaws.com/release1.pt"
    if args.weights_file is None:
      args.weights_file = os.path.expanduser(
        "~/.cache/omegafold_ckpt/model.pt"
      )
  elif args.model == 2:
    weights_url = "https://helixon.s3.amazonaws.com/release2.pt"
    if args.weights_file is None:
      args.weights_file = os.path.expanduser(
        "~/.cache/omegafold_ckpt/model2.pt"
      )
  else:
    raise ValueError(
      f"Model {args.model} is not available, "
      f"we only support model 1 and 2"
    )
  weights_file = args.weights_file
  if weights_file or weights_url:
    weights = _load_weights(weights_url, weights_file)
    weights = weights.pop('model', weights)
  else:
    weights = None

  forward_config = argparse.Namespace(
    subbatch_size=args.subbatch_size,
    num_recycle=args.num_cycle,
  )

  return args, weights, forward_config


def get_optimize_args(device, name, wildtype, seq_len, saved_dir, stability_thr=60):
  args = argparse.Namespace(
    device=device,
    name=name,
    wildtype=wildtype,
    seq_len=seq_len,
    done_cond = {
      'max_steps': 10,
      'max_target': 1.2,
      'max_mutation': 6,
    },
    stability_thr=stability_thr,
    saved_dir=saved_dir,
    total_steps=100000
  )
  return args