import re
import tiktoken

TOKENIZER_SPLIT_REGEX = r'([,.:;?!_"\'()]|--|\s)'
TOKENIZER_SUB_REGEX = r'\s+([,.?!"()\'])'
TOKENIZER_UNKNOWN_TOKEN = "<|unk|>"
TOKENIZER_END_OF_TEXT_TOKEN = "<|endoftext|>"
with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()
preprocessed = re.split(TOKENIZER_SPLIT_REGEX, raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
print("Number of tokens:", len(preprocessed))

unique_words = sorted(set(preprocessed))
unique_words.extend([TOKENIZER_END_OF_TEXT_TOKEN, TOKENIZER_UNKNOWN_TOKEN])
vocab_size = len(unique_words)
print("Vocabulary size:", vocab_size)

vocab = {token:integer for integer, token in enumerate(unique_words)}
for i, item in enumerate(vocab.items()):
  # print(item)
  if i >= 50:
    break

class Tokenizer:
  def __init__(self, vocab):
    self.str_to_int = vocab
    self.int_to_str = {i:s for s,i in vocab.items()}

  def encode(self, text):
    preprocessed = re.split(TOKENIZER_SPLIT_REGEX, text)
    preprocessed = [item for item in preprocessed if item.strip()]
    preprocessed = [item if item in self.str_to_int else TOKENIZER_UNKNOWN_TOKEN for item in preprocessed]
    ids = [self.str_to_int[s] for s in preprocessed]
    return ids

  def decode(self, ids):
    text = " ".join([self.int_to_str[i] for i in ids])
    text = re.sub(TOKENIZER_SUB_REGEX, r'\1', text)
    return text

tokenizer = tiktoken.get_encoding("gpt2")  
encoded_text = tokenizer.encode(raw_text)
print(len(encoded_text))