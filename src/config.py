from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
DATA_FEATURES   = ROOT / "data" / "features"

TRAIN_FASTA     = DATA_RAW / "Train.fasta"
TEST_FASTA      = DATA_RAW / "test.fasta"

TRAIN_PARSED    = DATA_PROCESSED / "train_parsed.csv"
TEST_PARSED     = DATA_PROCESSED / "test_parsed.csv"
CLASS_SUMMARY   = DATA_PROCESSED / "class_summary.csv"

TRAIN_PHYSCHEM  = DATA_FEATURES / "train_physchem.csv"
TEST_PHYSCHEM   = DATA_FEATURES / "test_physchem.csv"
TRAIN_SEQ       = DATA_FEATURES / "train_sequence.csv"
TEST_SEQ        = DATA_FEATURES / "test_sequence.csv"
TRAIN_PLM       = DATA_FEATURES / "train_plm.npy"
TEST_PLM        = DATA_FEATURES / "test_plm.npy"
TRAIN_FUSED     = DATA_FEATURES / "train_fused.csv"
TEST_FUSED      = DATA_FEATURES / "test_fused.csv"

# ── Results paths ─────────────────────────────────────────────────────────────
RESULTS         = ROOT / "results"
TABLES          = RESULTS / "tables"
FIGURES         = RESULTS / "figures"
LOGS            = RESULTS / "logs"

# ── Model settings ────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
CV_FOLDS        = 5
TEST_SIZE       = 0.2

# ── Feature settings ──────────────────────────────────────────────────────────
PLM_MODEL_NAME  = "facebook/esm2_t6_8M_UR50D"   # swap for a larger ESM2 if needed
PLM_BATCH_SIZE  = 32
MAX_SEQ_LEN     = 200

# ── CNN settings ──────────────────────────────────────────────────────────────
CNN_EPOCHS      = 50
CNN_BATCH_SIZE  = 32
CNN_LR          = 1e-3
