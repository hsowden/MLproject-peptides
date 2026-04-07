def parse_fasta(filepath):
    """
    Parse a FASTA file and return a list of (header, sequence) tuples.
    Handles multi-line sequences automatically.
    """
    records = []
    header = None
    seq_parts = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # Save the previous record before starting a new one
                if header is not None:
                    records.append((header, ''.join(seq_parts)))
                header = line[1:]  # Strip the '>'
                seq_parts = []
            else:
                seq_parts.append(line)

        # Don't forget the last record
        if header is not None:
            records.append((header, ''.join(seq_parts)))

    return records


# --- Usage ---
train_records = parse_fasta('/mnt/user-data/uploads/Train.fasta')
test_records  = parse_fasta('/mnt/user-data/uploads/test.fasta')

print(f"Train sequences: {len(train_records)}")
print(f"Test sequences:  {len(test_records)}")

# Inspect a record
header, seq = train_records[0]
print(f"\nHeader:   {header}")
print(f"Sequence: {seq}")
print(f"Length:   {len(seq)}")