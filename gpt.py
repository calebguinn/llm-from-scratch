import torch
from torch import nn as nn
import tiktoken

GPT_CONFIG_124M = {
  "vocab_size": 50257,      #Vocabulary Size
  "context_length": 1024,   #Context Length
  "emb_dim": 768,           #Embedding Dimension
  "n_heads": 12,            #Number of Attention Heads
  "n_layers": 12,           #Number of Layers
  "drop_rate": 0.1,         #Dropout Rate
  "qkv_bias": False         #Query-Key-Value Bias
}

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

class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(
      torch.sqrt(torch.tensor(2.0/torch.pi))*
      (x + 0.044715 * torch.pow(x,3))))

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

class TransformerBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attn = MultiHeadAttention(
      d_in=config["emb_dim"],
      d_out=config["emb_dim"],
      context_length=config["context_length"],
      num_heads=config["n_heads"],
      dropout=config["drop_rate"],
      qkv_bias=config["qkv_bias"]
    )
    self.ff = FeedForward(config)
    self.norm1 = LayerNorm(config["emb_dim"])
    self.norm2 = LayerNorm(config["emb_dim"])
    self.drop_shortcut = nn.Dropout(config["drop_rate"])

  def forward(self, x):
    shortcut = x
    x = self.norm1(x)
    x = self.attn(x)
    x = self.drop_shortcut(x)
    x = x + shortcut

    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut
    return x

class GPTModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
    self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
    self.drop_emb = nn.Dropout(config["drop_rate"])
    self.trf_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["n_layers"])])
    self.final_norm = LayerNorm(config["emb_dim"])
    self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
    x = tok_embeds + pos_embeds
    x = self.drop_emb(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
# print("Input Batch:", batch)
# print("Output Shape:", out.shape)
# print(out)

total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}") #163,009,536, larger due to weight tying

total_params_gpt2 = (
  total_params - sum(p.numel() for p in model.out_head.parameters())
)
# print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}") #124,412,160, the number of parameters in GPT 2

total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024*1024)
# print(f"Total size of the model: {total_size_mb:.2f} MB")