"""
train_llm.py — Minimal GPT-style Transformer from Scratch for Telugu
Author: Druva Kumar
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

# -------------------------------
# 1️⃣ Config
# -------------------------------
CONFIG = {
    "data_file": "Telugu_Tokenized_Data.json",  # pre-tokenized JSON list of token IDs
    "model_save_path": "llm_from_scratch.pt",
    "vocab_size": 25000,        # approximate vocab size
    "embedding_dim": 128,       # small for CPU
    "num_heads": 4,
    "num_layers": 2,
    "hidden_dim": 256,
    "sequence_length": 50,
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 0.002,
    "dropout": 0.1,
    "max_sequences": 50000      
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Training on device: {DEVICE}")

# -------------------------------
# 2️⃣ Dataset
# -------------------------------
class TeluguDataset(Dataset):
    def __init__(self, json_file, seq_len, max_sequences=None):
        with open(json_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Flatten sequences and keep only sequences longer than seq_len
        flat_sequences = []
        for seq in raw_data:
            if len(seq) > seq_len:
                flat_sequences.append(seq[:seq_len+1])
            if max_sequences and len(flat_sequences) >= max_sequences:
                break

        self.data = flat_sequences
        self.seq_len = seq_len
        print(f"✅ Loaded {len(self.data)} sequences for training")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

# -------------------------------
# 3️⃣ GPT-style Block
# -------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x

# -------------------------------
# 4️⃣ GPT-style Language Model
# -------------------------------
class GPTSmall(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, seq_len, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(seq_len, embedding_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.fc(x)
        return logits

# -------------------------------
# 5️⃣ Training
# -------------------------------
def train(model, dataloader, criterion, optimizer, epochs):
    model.to(DEVICE)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx+1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Avg Loss: {total_loss/(batch_idx+1):.4f}")

        print(f"✅ Epoch {epoch+1} completed, Avg Loss: {total_loss/len(dataloader):.4f}")

# -------------------------------
# 6️⃣ Main
# -------------------------------
def main():
    dataset = TeluguDataset(CONFIG["data_file"], CONFIG["sequence_length"], CONFIG["max_sequences"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    model = GPTSmall(
        vocab_size=CONFIG["vocab_size"],
        embedding_dim=CONFIG["embedding_dim"],
        num_heads=CONFIG["num_heads"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        seq_len=CONFIG["sequence_length"],
        dropout=CONFIG["dropout"]
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    print("🚀 Starting training...")
    train(model, dataloader, criterion, optimizer, CONFIG["epochs"])

    print(f"💾 Saving model to {CONFIG['model_save_path']}...")
    torch.save(model.state_dict(), CONFIG["model_save_path"])
    print("🎉 Training complete!")

if __name__ == "__main__":
    main()
