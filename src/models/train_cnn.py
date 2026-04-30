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

# The 20 standard amino acids; order determines integer index
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# Maps each amino acid character to a unique integer (1-20); 0 is reserved for padding
AA_TO_IDX   = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 = pad


def encode_sequence(seq: str, max_len: int = MAX_SEQ_LEN) -> list[int]:
    # Convert each character to its integer index; unknown chars map to 0 (pad)
    enc = [AA_TO_IDX.get(aa, 0) for aa in seq[:max_len]]
    # Pad with zeros on the right so every sequence is exactly max_len long
    enc += [0] * (max_len - len(enc))
    return enc


class PeptideCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int,
                 num_filters: int = 128, kernel_sizes=(3, 5, 7), dropout: float = 0.3):
        super().__init__()
        # Learnable embedding table: maps each amino acid index to a dense vector of size embed_dim
        # padding_idx=0 ensures pad tokens contribute nothing to gradients
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        # Three parallel 1D conv layers with different kernel sizes (3, 5, 7)
        # Each detects local sequence motifs at a different length scale
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes
        ])
        # Dropout for regularisation: randomly zeros 30% of neurons during training
        self.dropout = nn.Dropout(dropout)
        # Final classification layer: maps concatenated conv features → class scores
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x shape: (B, L) → embed → (B, L, E) → permute → (B, E, L) for Conv1d
        x = self.embedding(x).permute(0, 2, 1)   # (B, E, L)
        # Apply each conv + ReLU, then global max pool to get one value per filter
        # Max pool asks: "did this motif appear anywhere in the sequence?"
        pooled = [torch.relu(conv(x)).max(dim=-1).values for conv in self.convs]
        # Concatenate all three conv outputs → (B, num_filters * 3)
        x = torch.cat(pooled, dim=-1)
        x = self.dropout(x)
        # Raw class scores (logits); no softmax here — CrossEntropyLoss expects logits
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
    torch.manual_seed(RANDOM_SEED)  # Ensures reproducibility across runs
    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Encode string labels (e.g. "toxic"/"non-toxic") to integers (0/1)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)  # Fit on train, then transform
    y_test_enc  = le.transform(y_test)       # Transform test using same mapping
    num_classes = len(le.classes_)

    # Encode all sequences to fixed-length integer tensors
    X_train = torch.tensor([encode_sequence(s) for s in sequences_train], dtype=torch.long)
    X_test  = torch.tensor([encode_sequence(s) for s in sequences_test],  dtype=torch.long)
    y_tr    = torch.tensor(y_train_enc, dtype=torch.long)
    y_te    = torch.tensor(y_test_enc,  dtype=torch.long)

    # DataLoader handles batching and shuffling each epoch
    train_loader = DataLoader(TensorDataset(X_train, y_tr), batch_size=batch_size, shuffle=True)

    # Instantiate model and move to target device
    model = PeptideCNN(
        vocab_size=len(AMINO_ACIDS),
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_filters=num_filters,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adaptive learning rate optimiser
    criterion = nn.CrossEntropyLoss()                         # Standard loss for multi-class classification

    for epoch in range(1, epochs + 1):
        model.train()  # Enables dropout and batch norm (training mode)
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()          # Clear gradients from previous step
            loss = criterion(model(xb), yb)  # Forward pass + compute loss
            loss.backward()                # Backprop: compute gradients
            optimizer.step()               # Update weights
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}  loss={total_loss / len(train_loader):.4f}")

    # Evaluate on test set
    model.eval()  # Disables dropout for deterministic inference
    with torch.no_grad():  # No gradient tracking needed during evaluation
        logits = model(X_test.to(device))               # Raw class scores
        preds  = logits.argmax(dim=-1).cpu().numpy()    # Predicted class index
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()  # Class probabilities for AUC

    # Compute all metrics (accuracy, SN, SP, MCC, AUC) via shared evaluation module
    from src.evaluation.metrics import compute_metrics
    results = compute_metrics(y_test_enc, preds, probs, model_name="CNN")
    # Attach raw arrays so evaluate_all() can generate confusion matrices and ROC curves
    results["_y_true"] = y_test_enc
    results["_y_pred"] = preds
    results["_y_prob"] = probs
    return results, model, le