"""
Protein Language Model embeddings using ESM-2 (HuggingFace Transformers).

Requires:  pip install transformers torch
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from src.config import (
    TRAIN_PARSED, TEST_PARSED, TRAIN_PLM, TEST_PLM,
    PLM_MODEL_NAME, PLM_BATCH_SIZE, MAX_SEQ_LEN,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str = PLM_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_sequences(
    sequences: list[str],
    tokenizer,
    model,
    batch_size: int = PLM_BATCH_SIZE,
    max_len: int = MAX_SEQ_LEN,
    device=None,
) -> np.ndarray:
    if device is None:
        device = get_device()
    model = model.to(device)

    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len + 2,  # +2 for special tokens
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean-pool over sequence positions (exclude CLS/EOS tokens)
        hidden = outputs.last_hidden_state[:, 1:-1, :]
        mask   = inputs["attention_mask"][:, 1:-1].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(pooled.cpu().numpy())

        print(f"  Embedded {min(i + batch_size, len(sequences))}/{len(sequences)}", end="\r")

    print()
    return np.vstack(all_embeddings)


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PARSED)
    test  = pd.read_csv(TEST_PARSED)

    print(f"Loading model: {PLM_MODEL_NAME}")
    tokenizer, model = load_model()
    device = get_device()
    print(f"Device: {device}")

    print("Embedding train set...")
    train_emb = embed_sequences(train["sequence"].tolist(), tokenizer, model, device=device)
    print("Embedding test set...")
    test_emb  = embed_sequences(test["sequence"].tolist(),  tokenizer, model, device=device)

    np.save(TRAIN_PLM, train_emb)
    np.save(TEST_PLM,  test_emb)
    print(f"Train embeddings: {train_emb.shape}  → {TRAIN_PLM}")
    print(f"Test  embeddings: {test_emb.shape}   → {TEST_PLM}")
