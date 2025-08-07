
import torch
from timeit import default_timer as timer
from tqdm import tqdm
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from data.datasets.data_prep_utils import intra_doc_causal_mask, fast_intra_doc_causal_mask


path = "/fast/najroldi/data/lm/fwedu/fwedu_sample_10B_tokenizer_GPT2/ctx_2048/train"
train_set = load_from_disk(path)
max_seq_len = 2048+1

train_set[0]['docs_lengths']

# Test speed
time_v0 = 0.
time_v1 = 0.
n_trials = 10_000

for i in tqdm(range(n_trials)):
  chunk = train_set[i]
  
  time_start = timer()
  mask_v0 = intra_doc_causal_mask(train_set[0]['docs_lengths'], 2049, device='cuda')
  time_v0 += timer() - time_start
  
  time_start = timer()
  mask_v1 = fast_intra_doc_causal_mask(train_set[0]['docs_lengths'], 2049, device='cuda')
  time_v1 += timer() - time_start
  
  assert torch.equal(mask_v0, mask_v1)
  

print(f"time_v0 = {time_v0/n_trials}")
print(f"time_v1 = {time_v1/n_trials}")

# Let's visualize some masks
ii = 541
text = train_set[ii]['input_ids']
docs_lengths = train_set[ii]['docs_lengths']

tokenizer = tokenizer = AutoTokenizer.from_pretrained('gpt2')

tokenizer.decode(text)
mask = intra_doc_causal_mask(train_set[0]['docs_lengths'], 2049, device='cuda')
