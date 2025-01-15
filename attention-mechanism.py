import torch

def naive_softmax(x):
  return torch.exp(x) / torch.exp(x).sum(dim=0)

inputs = torch.tensor(
  [[0.43,0.15,0.89],
  [0.55,0.87,0.66],
  [0.57,0.85,0.64], 
  [0.22,0.58,0.33],
  [0.77,0.25,0.10],
  [0.05,0.80,0.55]]
)

query = inputs[1]
attention_scores = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
  attention_scores[i] = torch.dot(x_i, query)
print(attention_scores)
print("Attention weights:", attention_scores/attention_scores.sum())
print("Attention weight sum:", (attention_scores/attention_scores.sum()).sum())

attention_score_naive = naive_softmax(attention_scores)
print("Attention weights:", attention_score_naive)
print("Attention weight sum:", attention_score_naive.sum())

attention_weights = torch.softmax(attention_scores, dim=0)
query = inputs[1]
context_vec2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
  context_vec2 += attention_weights[i]*x_i
print(context_vec2)

attention_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
  for j, x_j in enumerate(inputs):
    attention_scores[i,j] = torch.dot(x_i,x_j)
# print(attention_scores)

attention_scores = inputs @ inputs.T
attention_weights = torch.softmax(attention_scores, dim=-1)
print(attention_scores)
print(attention_weights)

context_vectors = attention_weights @ inputs
print(context_vectors)