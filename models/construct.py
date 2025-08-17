
from fractions import Fraction
import wandb


def construct_model(cfg):
  """Initalize a model from config. Counts parameters."""

  # Transformer++
  if cfg.model == "transformer":
    from .transformer import Transformer, ModelConfig
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
    model = Transformer(model_cfg)

  # Pythia
  elif cfg.model.startswith("pythia"):
    from transformers import AutoConfig, AutoModelForCausalLM
    model_cfg = AutoConfig.from_pretrained(f"EleutherAI/{cfg.model}")
    model =  AutoModelForCausalLM.from_config(model_cfg) # NOTE: vocab_size=50304 here!
    model.init_weights() # explict init, since I am not sure it is done in 'from_config'
  
  else:
    raise NotImplementedError(f"Not implemented model: {cfg.model}.")
  
  if hasattr(model, 'count_params'):
    n_params = model.count_params(non_embedding=False)
    n_params_no_embed = model.count_params(non_embedding=True)
    print(f"Number of parameters: {n_params:_}")
    print(f"Number of non-embedding parameters: {n_params_no_embed:_}")
    if wandb.run is not None:
      wandb.log({
        "n_params": n_params,
        "n_params_no_embed": n_params_no_embed
      })
  
  return model, model_cfg

