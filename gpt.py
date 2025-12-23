import torch
from bpe import BPE
from model import GPTLanguageModel
from utils import get_batch, estimate_loss
from config import *

torch.manual_seed(1337)

with open("data/swift.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = BPE(text)
data_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

n = int(0.9 * len(data_ids))
data = {
    "train": data_ids[:n],
    "val": data_ids[n:]
}

model = GPTLanguageModel(tokenizer.vocab_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

for step in range(MAX_ITERS):
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss(model, data, DEVICE)
        print(f"Step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch(data, "train", DEVICE)
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(tokenizer.decode(model.generate(context, 2000)[0].tolist()))