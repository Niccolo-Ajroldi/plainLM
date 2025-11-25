"""
Compute top Hessian eigenvalues (and optional eigenvectors) for one checkpoint
or a list of checkpoints using PyHessian.

Loads the model/config from ckpt folder, builds a validation dataloader, runs Hessian
estimator, and saves results. Supports single .pth paths or files listing many.
Can run in parallel on a single file, uisng `row_idx` to assign rows (ckpts) to jobs..
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# torch/autograd/graph.py:823: UserWarning: Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient which can cause a memory leak. We recommend using autograd.grad when creating the graph to avoid this. If you have to use this function, make sure to reset the .grad fields of your parameters to None after use to break the cycle and avoid the leak. (Triggered internally at /pytorch/torch/csrc/autograd/engine.cpp:1260.)
#  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
                                                 
import pdb
import torch
import torch.nn as nn
from pathlib import Path
from absl import app, flags
from pyhessian import hessian
from fractions import Fraction
import json

from torch.utils.data import DataLoader, SequentialSampler
from datasets import Dataset, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM
from models.transformer_explicit_attention import TransformerExplicit, ModelConfig
import utils


if __name__ == "__main__":

  # Job specs.
  flags.DEFINE_string('paths', None, 'A file listing checkpoints to eval spectra on.')
  flags.DEFINE_integer('row_idx', -1, 'Row index.')
  flags.DEFINE_string('save_path', None, 'Path to save results.')
  flags.DEFINE_bool('save_vectors', False, 'Wheter to save eigenvectors or not.')

  # Dataloader specs.
  flags.DEFINE_integer('samples', 1, 'Number of samples in dataloader.')
  flags.DEFINE_integer('bsz', 1, 'Batch size.')

  # PyHessian Power Iteration specs default values.
  flags.DEFINE_integer('top_n', 1, 'Number of eigenvalues to compute.')
  flags.DEFINE_integer('max_iter', 100, 'Number of power iterations.')
  flags.DEFINE_float('tol', 1e-3, 'Poer iteration tolerance.')

# Parse flags.
FLAGS = flags.FLAGS


def build_dataloader(validset_path, samples, bsz):
  valid_set = load_from_disk(validset_path)
  if not isinstance(valid_set , Dataset):
    raise ValueError("dataset should be a datasets.Dataset")
  valid_set = valid_set.take(samples * bsz)
  
  def collate_fn(batch):
    return {"input_ids": torch.stack([x["input_ids"] for x in batch], dim=0)}

  validloader = DataLoader(
    valid_set,
    batch_size=bsz,
    num_workers=4,
    shuffle=False,
    sampler=SequentialSampler(valid_set),
    collate_fn=collate_fn
  )
  return validloader


def build_model(cfg):
  if cfg.model == "transformer":
    model_cfg = ModelConfig(
      vocab_size = cfg.vocab_size,
      dim = cfg.d_model,
      expand = float(Fraction(cfg.expand)),
      n_layers = cfg.n_layers,
      n_heads = cfg.n_heads,
      rmsorm_eps = 1e-6,
      mlp = cfg.mlp_class,
      seq_len = cfg.seq_len,
      tie_embeddings = cfg.tie_embeddings
    )
    model = TransformerExplicit(model_cfg)

  elif cfg.model.startswith("pythia"):
    model_cfg = AutoConfig.from_pretrained(f"EleutherAI/{cfg.model}")
    model_cfg._attn_implementation = "eager" # disable flash attention
    model =  AutoModelForCausalLM.from_config(model_cfg)
    model.init_weights() # explict init, since I am not sure it is done in 'from_config'

  return model


def spectra(model, dataloader, seq_len, samples, max_iter, tol, top_n):
  """Compute spectra. Currently only working in cuda."""
  model.eval()

  if samples == 1:
    batch = next(iter(dataloader))
    inputs = batch['input_ids'][:,:seq_len].to('cuda')
    targets = batch['input_ids'][:,1:(seq_len+1)].to('cuda')
    data = (inputs, targets)
    dataloader_hessian = None
  else:
    data = None
    dataloader_hessian = dataloader

  hessian_comp = hessian(
    model,
    criterion=nn.CrossEntropyLoss(),
    data=data,
    dataloader=dataloader_hessian,
    precond=None,
    cuda=True
  )
  top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(maxIter=max_iter, tol=tol, top_n=top_n)

  model.zero_grad(set_to_none=True)

  return top_eigenvalues, top_eigenvectors


def main(_):

  device = f'cuda:0'
  torch.cuda.device(device)

  # Load checkpoint list
  p = Path(FLAGS.paths)

  # Single path to ckpt or file containing a list of ckpt paths.
  if p.is_file() and p.suffix == ".pth":
    ckpt_list = [p]
  else:
    ckpt_list = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                ckpt_list.append(Path(line))

  # Enforce row_idx rules
  if len(ckpt_list) == 1:
    if FLAGS.row_idx > 0:
      raise ValueError("row_idx must be -1 or 0 when a single checkpoint is given.")

  # Mode: parallel (row_idx>=0) or sequential (row_idx==-1).
  if FLAGS.row_idx == -1:
      row_indices = range(len(ckpt_list))
  else:
      row_indices = [FLAGS.row_idx]

  # Load a config.
  cfg_path = ckpt_list[row_indices[0]].parent / "config.yaml"
  print(f"Loading config from {cfg_path}")
  cfg, _ = utils.load_config(cfg_path)

  # Dataloader.
  validloader = build_dataloader(validset_path=cfg.validset_path, samples=FLAGS.samples, bsz=FLAGS.bsz)
  print(f"Number of validation batches: {len(validloader)}")
  print(f"Batch size: {validloader.batch_size}")
  
  # Model: either Trnasformer or Pythia
  model = build_model(cfg)
  model.to(device)
  print(model)

  # Loop over checkpoints, or not.
  for row_idx in row_indices:

    ckpt_file = ckpt_list[row_idx]

    print(ckpt_file.name)
    if not ckpt_file.exists():
      print(f"\tCheckpoint file {ckpt_file} does not exist!")

    # Load checkpoint
    print(f"\tLoading checkpoint from {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location=device)

    # Recover step
    step = ckpt['step'] if 'step' in ckpt else None

    # Manipulate the saved state_dict, to allow loading LAWA.
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Load params into model
    # NOTE: this will fail when the model is `TransformerExplicit` loading from `Transformer`, because of one extra buffer
    model.load_state_dict(state_dict)
    model.zero_grad()
    print(f"\tState dictionary succesfully loaded into model!")

    # Sharpness
    print(f"\tComputing sharpness")
    top_eigenvalues, top_eigenvectors = spectra(
        model, 
        validloader, 
        cfg.seq_len, 
        FLAGS.samples,
        FLAGS.max_iter,
        FLAGS.tol,
        FLAGS.top_n,
    )

    if FLAGS.save_path:
      scalars = {
        'ckpt_path': str(ckpt_file),
        'step': step,
        'top_eigenvalues': top_eigenvalues,
      }
      scalars_path = Path(FLAGS.save_path) / f"row_{row_idx}" / f'scalars.json'
      scalars_path.parent.mkdir(parents=True, exist_ok=True)
      with open(scalars_path, "w") as f:
        json.dump(scalars, f, indent=2)
      print(f"\tScalar metrics saved to {scalars_path}")

      if FLAGS.save_vectors:
        tensors = {'top_eigenvectors': top_eigenvectors}
        tensors_path = Path(FLAGS.save_path) / f"row_{row_idx}" / f'tensors.pth'
        tensors_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, tensors_path)
        print(f"\tTensor saved to {tensors_path}")

  print('Finished!')


if __name__ == "__main__":
  app.run(main)
