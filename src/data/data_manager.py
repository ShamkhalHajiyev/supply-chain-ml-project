import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import kagglehub
from kagglehub import KaggleDatasetAdapter

# ========================
# PATHS & CONFIG
# ========================

# Base directory (two levels above this file)
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Kaggle dataset info
DATASET_ID = "saicharankomati/dataco-supply-chain-dataset"
RAW_FILENAME = "DataCoSupplyChainDataset.csv"
RAW_FILE = RAW_DIR / RAW_FILENAME

# Refresh interval (24h)
MAX_AGE_SECONDS = 24 * 60 * 60


def timestamp():
    """Generate timestamp for versioning."""
    return datetime.now().strftime("%Y%m%d_%H%M")


# ========================
# FILE AGE CHECK
# ========================

def file_is_old(path: Path, max_age: int) -> bool:
    """Return True if file missing or older than max_age."""
    if not path.exists():
        return True
    age = time.time() - path.stat().st_mtime
    return age > max_age


# ========================
# DOWNLOAD RAW DATA
# ========================

def download_raw():
    print(f"📥 Downloading dataset from Kaggle: {DATASET_ID}")

    # Try loading with Latin-1 encoding (handles special chars)
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_ID,
        RAW_FILENAME,
        pandas_kwargs={"encoding": "latin1"}
    )

    df.to_csv(RAW_FILE, index=False, encoding="latin1")
    print(f"✅ Download complete → saved to {RAW_FILE}")
    return df


# ========================
# LOAD RAW DATA
# ========================

def load_raw(filename: str = RAW_FILENAME) -> pd.DataFrame:
    """Load raw CSV; auto-download if missing or expired."""
    file_path = RAW_DIR / filename

    if file_is_old(file_path, MAX_AGE_SECONDS):
        print("⚠️ Raw dataset missing or outdated — downloading fresh copy...")
        return download_raw()

    print(f"📂 Loading raw file: {file_path}")

    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        print("⚠️ UTF-8 decode failed. Retrying with Latin-1...")
        return pd.read_csv(file_path, encoding="latin1")


# ========================
# SAVE FUNCTIONS
# ========================

def save_interim(df: pd.DataFrame, name: str) -> Path:
    filename = f"{name}_{timestamp()}.parquet"
    path = INTERIM_DIR / filename
    df.to_parquet(path, index=False)
    print(f"✅ Interim saved → {path}")
    return path


def save_processed(df: pd.DataFrame, name: str) -> Path:
    filename = f"{name}_{timestamp()}.parquet"
    path = PROCESSED_DIR / filename
    df.to_parquet(path, index=False)
    print(f"✅ Processed saved → {path}")
    return path


# ========================
# LOAD LATEST FILE
# ========================

def load_latest(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"), reverse=True)
    if not files:
        raise FileNotFoundError(f"❌ No parquet files in {directory}")
    print(f"📂 Loading latest file: {files[0]}")
    return pd.read_parquet(files[0])


def load_latest_interim():
    return load_latest(INTERIM_DIR)


def load_latest_processed():
    return load_latest(PROCESSED_DIR)