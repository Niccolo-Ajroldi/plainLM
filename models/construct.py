from fractions import Fraction

import wandb


def construct_model(cfg):
  """Initalize a model from config. Counts parameters."""

  # Transformer++
  if cfg.model == "transformer":
    from .transformer import ModelConfig, Transformer

    model_cfg = ModelConfig(
      vocab_size=cfg.vocab_size,
      dim=cfg.d_model,
      expand=float(Fraction(cfg.expand)),
      n_layers=cfg.n_layers,
      n_heads=cfg.n_heads,
      rmsnorm_eps=1e-6,
      mlp=cfg.mlp_class,
      seq_len=cfg.seq_len,
      tie_embeddings=cfg.tie_embeddings,
    )
    model = Transformer(model_cfg)

  # Pythia
  elif cfg.model.startswith("pythia"):
    from transformers import AutoConfig, AutoModelForCausalLM

    model_cfg = AutoConfig.from_pretrained(f"EleutherAI/{cfg.model}")
    model = AutoModelForCausalLM.from_config(model_cfg)  # NOTE: vocab_size=50304 here!
    model.init_weights()  # explict init, since I am not sure it is done in 'from_config'

  else:
    raise NotImplementedError(f"Not implemented model: {cfg.model}.")

  if hasattr(model, "count_params"):
    n_params = model.count_params(non_embedding=False)
    n_params_no_embed = model.count_params(non_embedding=True)
    print(f"Number of parameters: {n_params:_}")
    print(f"Number of non-embedding parameters: {n_params_no_embed:_}")
    if wandb.run is not None:
      wandb.log({"n_params": n_params, "n_params_no_embed": n_params_no_embed})

  return model, model_cfg


def get_param_groups(model, cfg):
  """
  Create param groups for a Transformer model.
  Bias and normalization layers are excluded from weight decay.
  """

  # filter out parameters that do not require grad
  named_param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
  param_names = named_param_dict.keys()

  # normalization layers
  norm_param_names = [n for n in param_names if "norm" in n]
  norm_params = [p for n, p in named_param_dict.items() if n in norm_param_names]

  # bias
  bias_param_names = [n for n in param_names if "bias" in n and n not in norm_param_names]
  bias_params = [p for n, p in named_param_dict.items() if n in bias_param_names]

  # all the ohers params
  norm_and_bias_names = norm_param_names + bias_param_names
  other_param_names = [n for n in param_names if n not in norm_and_bias_names]
  other_params = [p for n, p in named_param_dict.items() if n in other_param_names]

  # assemble param groups
  param_groups = [
    dict(
      params=norm_params,
      weight_decay=0.0,
    ),
    dict(
      params=bias_params,
      weight_decay=0.0,
    ),
    dict(
      params=other_params,
      weight_decay=cfg.weight_decay,
    ),
  ]

  # # sanity check
  # print("bias_param_names:\n\t" + "\n\t".join(bias_param_names))
  # print("norm_param_names:\n\t" + "\n\t".join(norm_param_names))
  # print("other_param_names:\n\t" + "\n\t".join(other_param_names))

  return param_groups
