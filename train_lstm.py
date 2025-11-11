"""
train_lstm.py — Minimal Telugu LSTM for Fast CPU Fine-Tuning
Author: Druva Kumar
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# 1️⃣ Config
# -------------------------------
CONFIG = {
    "data_file": "Telugu_Tokenized_Data.json",
    "model_path": "telugu_lstm_fast.pt",
    "embedding_dim": 100,        # smaller for faster CPU training
    "hidden_dim": 200,
    "num_layers": 2,
    "dropout": 0.2,
    "sequence_length": 50,      # shorter sequences -> faster
    "batch_size": 256,           # bigger batch -> fewer iterations
    "epochs": 3,                # quick training for fine-tuning
    "learning_rate": 0.004,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Training on device: {DEVICE}")

# -------------------------------
# 2️⃣ Dataset
# -------------------------------
class TeluguDataset(Dataset):
    def __init__(self, json_file, seq_len):
        self.seq_len = seq_len
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Keep only sequences long enough
        self.data = [seq[:seq_len + 1] for seq in self.data if len(seq) > seq_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

# -------------------------------
# 3️⃣ Simple LSTM Model
# -------------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# -------------------------------
# 4️⃣ Training Function
# -------------------------------
def train(model, dataloader, criterion, optimizer, epochs):
    model.to(DEVICE)
    for epoch in range(epochs):
        total_loss = 0
        hidden = None
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            output, hidden = model(x, hidden)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            # Detach hidden state to prevent backprop through entire history
            hidden = tuple([h.detach() for h in hidden])

            total_loss += loss.item()

            # Print progress every 10 batches
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Avg Loss: {total_loss/(i+1):.4f}")

# -------------------------------
# 5️⃣ Main
# -------------------------------
def main():
    dataset = TeluguDataset(CONFIG["data_file"], CONFIG["sequence_length"])
    if len(dataset) == 0:
        print("⚠️ No valid sequences found.")
        return

    # Compute vocab size
    vocab_size = max([token for seq in dataset.data for token in seq]) + 1
    print(f"Vocabulary size: {vocab_size}")
    print(f"Total sequences: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    model = LSTMLanguageModel(
        vocab_size,
        CONFIG["embedding_dim"],
        CONFIG["hidden_dim"],
        CONFIG["num_layers"],
        CONFIG["dropout"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    print("🚀 Starting training...")
    train(model, dataloader, criterion, optimizer, CONFIG["epochs"])
    print("🎯 Training completed!")

    torch.save(model.state_dict(), CONFIG["model_path"])
    print(f"💾 Model saved to {CONFIG['model_path']}")

if __name__ == "__main__":
    main()
