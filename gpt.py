import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
BATCH_SIZE = 64 # How many independent sequences will we process in parallel?
BLOCK_SIZE = 256 # What is the maximum context length for predictions?
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
LR = 3e-4
N_EMBED = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.manual_seed(1337)

with open('data/swift.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers (tokenize)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # Encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # Decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i: i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1: i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    model.train()
    
    return out

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out
    
class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        
        return out
    
class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBED) # Final layer norm
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (Batch, Time) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (Batch, Time, Channel). In this case Batch = BATCH_SIZE, Time = BLOCK_SIZE, Channel = vocab_size
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last BLOCK_SIZE tokens
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Get the predictions
            logits, _ = self(idx_cond)

            # Focus on the last time step
            logits = logits[:, -1, :] # Becomes (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # Sample from the distribution
            next_idx = torch.multinomial(probs, num_samples=1) # (B, 1)

            # Append sampled index to the sequence
            idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)
        
        return idx

model = GPTLanguageModel()
m = model.to(device)

#Print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'Model Parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=LR)

for iter in range(MAX_ITERS):
    # Every once in a while evaluate the loss on train and val sets
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))