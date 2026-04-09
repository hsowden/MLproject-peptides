"""
1-D CNN for peptide toxicity classification.
Requires: pip install torch
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from src.config import RANDOM_SEED, CNN_EPOCHS, CNN_BATCH_SIZE, CNN_LR, MAX_SEQ_LEN

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX   = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 = pad


def encode_sequence(seq: str, max_len: int = MAX_SEQ_LEN) -> list[int]:
    enc = [AA_TO_IDX.get(aa, 0) for aa in seq[:max_len]]
    enc += [0] * (max_len - len(enc))
    return enc


class PeptideCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int,
                 num_filters: int = 128, kernel_sizes=(3, 5, 7), dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)   # (B, E, L)
        pooled = [torch.relu(conv(x)).max(dim=-1).values for conv in self.convs]
        x = torch.cat(pooled, dim=-1)
        x = self.dropout(x)
        return self.fc(x)


def train_cnn(
    sequences_train, y_train,
    sequences_test,  y_test,
    embed_dim: int = 64,
    num_filters: int = 128,
    epochs: int = CNN_EPOCHS,
    batch_size: int = CNN_BATCH_SIZE,
    lr: float = CNN_LR,
) -> dict:
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    num_classes = len(le.classes_)

    X_train = torch.tensor([encode_sequence(s) for s in sequences_train], dtype=torch.long)
    X_test  = torch.tensor([encode_sequence(s) for s in sequences_test],  dtype=torch.long)
    y_tr    = torch.tensor(y_train_enc, dtype=torch.long)
    y_te    = torch.tensor(y_test_enc,  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_tr), batch_size=batch_size, shuffle=True)

    model = PeptideCNN(
        vocab_size=len(AMINO_ACIDS),
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_filters=num_filters,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}  loss={total_loss / len(train_loader):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        preds  = logits.argmax(dim=-1).cpu().numpy()
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()

    from src.evaluation.metrics import compute_metrics
    results = compute_metrics(y_test_enc, preds, probs, model_name="CNN")
    results["_y_true"] = y_test_enc
    results["_y_pred"] = preds
    results["_y_prob"] = probs
    return results, model, le
