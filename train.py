import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

## --------超参设置--------
batch_size = 16
block_size = 32
max_iters = 8000
eval_interval = 100
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
torch.manual_seed(9527)

## --------数据加载--------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
print(f"数据集字符总数: {len(text)}")
print(f"前 1000 个字符: {text[:1000]}")
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"字符集大小: {vocab_size}")
print("文本中出现的字符：", "".join(chars))
stoi = {ch: i for i, ch in enumerate(chars)}  # 字符到索引的映射
itos = {i: ch for i, ch in enumerate(chars)}  # 索引到字符的映射
encode = lambda s: [stoi[c] for c in s]  # 编码函数
decode = lambda l: "".join([itos[i] for i in l])  # 解码函数
print("编码结果：", encode("hello world"))
print("解码结果：", decode(encode("hello world")))

## --------数据处理--------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


## --------模型定义--------
class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * (C**-0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.proj(out)  # (B, T, n_embd)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.sa(self.ln1(x)))  # Self-attention
        x = x + self.dropout(self.ffwd(self.ln2(x)))  # Feed-forward
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # (B, T)
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.size()
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx


## --------模型训练--------
model = BigramLanguageModel(vocab_size, n_embd, n_head, n_layer, dropout)
model.to(device)
print(f"模型参数总数: {sum(p.numel() for p in model.parameters())/1e6}M")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_losses = []
val_losses = []
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_loss = losses["train"]
        val_loss = losses["val"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Iter {iter}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    X, Y = get_batch("train")
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.savefig("loss_plot.png")
plt.show()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=2000)
print("生成的文本：", decode(generated[0].tolist()))
torch.save(model.state_dict(), "nanoGPT.pth")
print("✅ 模型已保存: nanoGPT.pth")
print("🚀 训练完成!")
