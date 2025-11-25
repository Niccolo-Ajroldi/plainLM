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

# # 29_928_192 / 49_234_560 ##
# n_layers = 6
# d = 384
# n_heads = 6
# d_head = 64

# # 50_916_352 / 76_658_176 ##
# n_layers = 8
# d = 512
# n_heads = 8
# d_head = 64

# ?? / 115_158_400 ## 
n_layers = 10
d = 640
# n_heads = 10
# d_head = 64

# # 123_566_592 / 162_179_328 ##
# n_layers = 12
# d = 768
# n_heads = 12
# d_head = 64

# # 214_183_680 / 262_449_600 ##
# n_layers = 15
# d = 960
# n_heads = 15
# d_head = 64

# # 344_616_192 / 402_535_296 ## 
# n_layers = 18
# d = 1152
# n_heads = 18
# d_head = 64

vocab_size = 50277
n_params = tot_params(n_layers, d, vocab_size, weight_tying=True)
print(f"{int(n_params):_}")
