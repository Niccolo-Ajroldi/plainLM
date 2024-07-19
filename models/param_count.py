"""A script to count number of parameters in a transformer."""

multiple_of = 256  # mlp/glu inner dim is a multiple of this

def attn(d):
  return 4 * d**2

def glu(d):
  hidden_dim = d*8/3
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return 3 * d * hidden_dim

def mlp(d):
  hidden_dim = d*4
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
  return 2 * d * hidden_dim

def rmsnorm(d):
  return d

def block_params(d):
  return attn(d) + mlp(d) + 2*rmsnorm(d)
  # return attn(d) + glu(d) + 2*rmsnorm(d)

def tot_params(n_layers, d, vocab_size, weight_tying=True):
  nparams =  n_layers * block_params(d) + d*vocab_size +d
  if not weight_tying:
    nparams += d*vocab_size
  return nparams

d = 1536
n_layers = 24
vocab_size = 50280
n_params = tot_params(n_layers, d, vocab_size, weight_tying=False)
print(f"{int(n_params):_}")
