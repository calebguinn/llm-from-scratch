import torch
from torch import nn as nn

torch.manual_seed(123)
batch_example = torch.randn(2,5)
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)
variance = out.var(dim=-1, keepdim=True)
print("Mean: ", mean)
print("Variance: ", variance)

#Normalizing creates faster convergence to optimal weight values
out_norm = (out - mean) / torch.sqrt(variance)
mean = out_norm.mean(dim=-1, keepdim=True)
variance = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:", out_norm)
torch.set_printoptions(sci_mode=False)
print("Mean:", mean) #[0,0]
print("Variance:", variance) #[1,1]

class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5 #epsilon
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * norm_x + self.shift

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True)
print("Mean:", mean)
print("Variance:", variance)