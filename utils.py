import torch
from config import BATCH_SIZE, BLOCK_SIZE, EVAL_ITERS

def get_batch(data, split, device):
    src = data["train"] if split == "train" else data["val"]
    ix = torch.randint(len(src) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([src[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([src[i + 1:i + BLOCK_SIZE + 1] for i in ix])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, device):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data, split, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out