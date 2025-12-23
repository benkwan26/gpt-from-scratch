import torch

# Training
BATCH_SIZE = 64 # How many independent sequences will we process in parallel?
BLOCK_SIZE = 256 # What is the maximum context length for predictions?
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
LR = 3e-4

# Model
N_EMBED = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
VOCAB_SIZE = 500 # Desired final vocabulary size

# Tokenization
PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""

# Device
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'