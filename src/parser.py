import csv
from pathlib import Path
from src.config import TRAIN_FASTA, TEST_FASTA, TRAIN_PARSED, TEST_PARSED, CLASS_SUMMARY


def parse_fasta(filepath: Path) -> list[tuple[str, str, str]]:
    """
    Parse a FASTA file.

    Expected header format:  >ID|ClassName  (pipe-delimited)
    Returns a list of (id, class_label, sequence) tuples.
    Handles multi-line sequences automatically.
    """
    records = []
    header = None
    seq_parts = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(_parse_header(header, seq_parts))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)

    if header is not None:
        records.append(_parse_header(header, seq_parts))

    return records


def _parse_header(header: str, seq_parts: list[str]) -> tuple[str, str, str]:
    sequence = "".join(seq_parts)
    parts = header.split("|")
    seq_id = parts[0].strip()
    label = parts[1].strip() if len(parts) > 1 else "unknown"
    return seq_id, label, sequence


def records_to_csv(records: list[tuple[str, str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "sequence"])
        writer.writerows(records)
    print(f"Saved {len(records)} records → {out_path}")


def build_class_summary(train_records, test_records, out_path: Path) -> None:
    from collections import Counter
    train_counts = Counter(r[1] for r in train_records)
    test_counts  = Counter(r[1] for r in test_records)
    all_labels   = sorted(set(train_counts) | set(test_counts))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "train_count", "test_count"])
        for lbl in all_labels:
            writer.writerow([lbl, train_counts.get(lbl, 0), test_counts.get(lbl, 0)])
    print(f"Class summary → {out_path}")


if __name__ == "__main__":
    train = parse_fasta(TRAIN_FASTA)
    test  = parse_fasta(TEST_FASTA)
    records_to_csv(train, TRAIN_PARSED)
    records_to_csv(test,  TEST_PARSED)
    build_class_summary(train, test, CLASS_SUMMARY)
    print(f"\nTrain: {len(train)}  Test: {len(test)}")
    print("Sample:", train[0])
