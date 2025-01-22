import torch

class SelfAttention(nn.Module):
  def __init__(self, d_in, d_out, qkv_bias=False):
    super().__init__()
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

  def forward(self, x):
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)
    attention_scores = queries @ keys.T # omega
    attention_weights = torch.softmax(
      attention_scores / keys.shape[-1]**0.5, dim = -1
    )
    context_vector = attention_weights @ values
    return context_vector

inputs = torch.tensor(
  [[0.43,0.15,0.89],
  [0.55,0.87,0.66],
  [0.57,0.85,0.64], 
  [0.22,0.58,0.33],
  [0.77,0.25,0.10],
  [0.05,0.80,0.55]]
)

d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(789)
sa_v1 = SelfAttention(d_in, d_out)
# print(sa_v1(inputs))

queries = sa_v1.W_query(inputs)
keys = sa_v1.W_key(inputs)
attention_scores = queries @ keys.T
attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
# print(attention_weights)

context_length = attention_scores.shape[0]
simple_mask = torch.tril(torch.ones(context_length, context_length))
# print(simple_mask)
masked_scores = attention_weights*simple_mask #for implementing causal attention mask
# print(masked_scores)

#renormalize attention weights following mask
row_sums = masked_scores.sum(dim=-1, keepdim=True)
masked_scores_norm = masked_scores / row_sums
# print(masked_scores_norm)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attention_scores.masked_fill(mask.bool(), -torch.inf)
# print(masked)

attention_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
# print(attention_weights)

#dropout example, used to reduce overfitting remaining entries are scaled up 1/n where n is dropout rate
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6,6)
# print(dropout(example))

print(dropout(attention_weights))