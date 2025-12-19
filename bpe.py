# Hyperparameters
VOCAB_SIZE = 276 # Desired final vocabulary size

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokens = text.encode('utf-8')
tokens = list(map(int, tokens))

def count_pairs(tokens):
    counts = {}
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    
    return counts

def merge_pairs(tokens, pair, new_token):
    # In the list tokens, replace all consecutive occurrences of pair with new_token
    new_tokens = []
    i = 0
    while i < len(tokens):
        # If we are not at the very last position AND the pair matches, replace it
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    return new_tokens

num_merges = VOCAB_SIZE - 256
ids = list(tokens)

merges = {} # (int, int) -> int
for i in range(num_merges):
    pair_counts = count_pairs(ids)
    top_pair = max(pair_counts, key=pair_counts.get)
    idx = 256 + i
    ids = merge_pairs(ids, top_pair, idx)
    merges[top_pair] = idx

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    # Given ids (list of integers), return a Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode('utf-8', errors='replace')

    return text

def encode(text):
    # Given a string, return a list of integers (the tokens)
    tokens = list(text.encode('utf-8'))

    while len(tokens) >= 2:
        pair_counts = count_pairs(tokens)
        pair = min(pair_counts, key=lambda p: merges.get(p, float('inf')))

        if pair not in merges:
            break # Nothing else can be merged

        idx = merges[pair]
        tokens = merge_pairs(tokens, pair, idx)
    
    return tokens