import torch
from torch import nn as nn

class CausalAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):
    super().__init__()
    self.d_out = d_out
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length), diagonal=1)) #moves buffers to CPU or GPU

  def forward(self, x):
    b, num_tokens, d_in = x.shape #batch size, num tokens, input dimensions
    queries = self.W_query(x)
    keys = self.W_key(x)
    values = self.W_value(x)

    attention_scores = queries @ keys.transpose(1,2)
    attention_scores.masked_fill_(self.mask.bool() [:num_tokens, :num_tokens], -torch.inf)
    attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
    context_vector = attention_weights @ values
    return context_vector

class MultiHeadAttentionWrapper(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

  def forward(self, x):
    return torch.cat([head(x) for head in self.heads], dim=-1)


inputs = torch.tensor(
  [[0.43,0.15,0.89],
  [0.55,0.87,0.66],
  [0.57,0.85,0.64], 
  [0.22,0.58,0.33],
  [0.77,0.25,0.10],
  [0.05,0.80,0.55]]
)
batch = torch.stack((inputs,inputs), dim=0) #two inputs with six tokens each, embedding dimension of 3
torch.manual_seed(123)
context_length = batch.shape[1] #number of tokens
d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vectors = mha(batch)

# print(context_vectors)
# print("context_vectors.shape: ", context_vectors.shape)

class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert(d_out % num_heads == 0), "d_out must be divisible by num_heads"
    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

  def forward(self, x):
    b, num_tokens, d_in = x.shape
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)

    keys = keys.transpose(1,2)
    queries = queries.transpose(1,2)
    values = values.transpose(1,2)

    attention_scores = queries @ keys.transpose(2,3)
    mask_bool = self.mask.bool() [:num_tokens, :num_tokens]

    attention_scores.masked_fill_(mask_bool, -torch.inf)

    attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
    attention_weights = self.dropout(attention_weights)

    context_vector = (attention_weights @ values).transpose(1,2)
    context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
    context_vector = self.out_proj(context_vector)
    return context_vector

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vectors = mha(batch)
print(context_vectors)
print(context_vectors.shape)
