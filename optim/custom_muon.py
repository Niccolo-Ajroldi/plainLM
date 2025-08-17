

def get_param_groups_muon(model, cfg):
  """Create param groups for Muon and Dion."""

  # filter out parameters that do not require grad
  named_param_dict = {n: p for n,p in model.named_parameters() if p.requires_grad}
  param_names = named_param_dict.keys()

  # normaliz
  norm_param_names = [n for n in param_names if "norm" in n]
  norm_params = [p for n, p in named_param_dict.items() if n in norm_param_names]

  # bias
  bias_param_names = [n for n in param_names if "bias" in n and n not in norm_param_names]
  bias_params = [p for n, p in named_param_dict.items() if n in bias_param_names]

  # embedding
  embed_param_names = [n for n in param_names if "embed_tokens" in n]
  embed_params = [p for n, p in named_param_dict.items() if n in embed_param_names]

  # lm head
  lm_head_param_names = [n for n in param_names if "lm_head" in n]
  lm_head_params = [p for n, p in named_param_dict.items() if n in lm_head_param_names]

  # all the ohers should be 2D tensors
  non_matrix_param_names = bias_param_names + norm_param_names + embed_param_names + lm_head_param_names
  matrix_param_names = [n for n in param_names if n not in non_matrix_param_names]
  matrix_params = [p for n, p in named_param_dict.items() if n in matrix_param_names]

  # assemble param groups
  param_groups = [
    dict(params=bias_params,    algorithm=cfg.optim_backup, weight_decay=0.0),
    dict(params=norm_params,    algorithm=cfg.optim_backup, weight_decay=0.0),
    dict(params=embed_params,   algorithm=cfg.optim_backup, weight_decay=cfg.weight_decay),
    dict(params=lm_head_params, algorithm=cfg.optim_backup, weight_decay=cfg.weight_decay),
    dict(params=matrix_params,  algorithm=cfg.optim, weight_decay=cfg.weight_decay), # TODO: remove wd on matrices?
    
  ]

  # sanity check
  # print("bias_param_names:\n\t" + "\n\t".join(bias_param_names))
  # print("norm_param_names:\n\t" + "\n\t".join(norm_param_names))
  # print("embed_param_names:\n\t" + "\n\t".join(embed_param_names))
  # print("lm_head_param_names:\n\t" + "\n\t".join(lm_head_param_names))
  # print("matrix_param_names:\n\t" + "\n\t".join(matrix_param_names))

  return param_groups
