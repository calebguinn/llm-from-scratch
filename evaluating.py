import tiktoken
import torch
from components import GPTModel, generate_text_simple

GPT_CONFIG_124M = {
  "vocab_size": 50257,      #Vocabulary Size
  "context_length": 256,   #Context Length
  "emb_dim": 768,           #Embedding Dimension
  "n_heads": 12,            #Number of Attention Heads
  "n_layers": 12,           #Number of Layers
  "drop_rate": 0.1,         #Dropout Rate
  "qkv_bias": False         #Query-Key-Value Bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference

