import torch
from torch import nn as nn
import matplotlib.pyplot as plt

class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.44715*torch.pow(x,3))))

gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8,3))
for i, (y,label) in enumerate(zip([y_gelu, y_relu],["GELU", "ReLU"]),1):
  plt.subplot(1,2,i)
  plt.plot(x,y)
  plt.title(f"{label} activation function")
  plt.xlabel("x")
  plt.ylabel(f"{label}(x)")
  plt.grid(True)
plt.tight_layout()
# plt.show()
#GELU is smoother and allows for small negative values. This means that more neurons can contribute to learning and optimization can be easier for deep networks

#feed forward brings the linear layer into higher dimensional space, allowing for exploration of richer representation space and greater generalization in training tasks
class FeedForward(nn.Module): 
  def __init__(self, config):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
      GELU(),
      nn.Linear(4 * config["emb_dim"], config["emb_dim"])
    )
  
  def forward(self, x):
    return self.layers(x)

GPT_CONFIG_124M = {
  "vocab_size": 50257,      #Vocabulary Size
  "context_length": 1024,   #Context Length
  "emb_dim": 768,           #Embedding Dimension
  "n_heads": 12,            #Number of Attention Heads
  "n_layers": 12,           #Number of Layers
  "drop_rate": 0.1,         #Dropout Rate
  "qkv_bias": False         #Query-Key-Value Bias
}

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2,3,768)
out = ffn(x)
print(out.shape)