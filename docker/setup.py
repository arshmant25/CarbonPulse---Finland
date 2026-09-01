# setup.py
# ─────────────────────────────────────────────────────────────
# Run this ONCE before `docker compose up` to:
#   1. Create the data/ directory
#   2. Copy your trained model files into data/
#   3. Validate all required files are present
#   4. Check the .env file exists
#
# Usage:
#   python setup.py
#   python setup.py --model-dir "C:\path\to\outputs_15min"
#   python setup.py --db-path "C:\path\to\project_data.db"

import sys
import shutil
import argparse
from pathlib import Path

# ── Default paths — adjust to your machine ─────────────────────
DEFAULT_MODEL_DIR = Path("outputs_15min")
DEFAULT_DB_PATH   = Path("data/project_data.db")
DATA_DIR          = Path("data")

# Files that MUST exist in model dir before Docker can run
REQUIRED_MODEL_FILES = [
    "gru_model.pt",
    "scaler_X_15min.pkl",
    "scaler_y_15min.pkl",
    "column_config.json",
]

# Optional but useful
OPTIONAL_MODEL_FILES = [
    "lstm_model.pt",
    "rnn_model.pt",
    "gru_results.json",
    "lstm_results.json",
    "rnn_results.json",
]


def check_env():
    env_file = Path(".env")
    example  = Path(".env.example")
    if not env_file.exists():
        if example.exists():
            shutil.copy(example, env_file)
            print("  Created .env from .env.example")
            print("  ⚠  Edit .env and set FINGRID_API_KEY before running Docker")
        else:
            print("  ⚠  No .env file found — create one with:")
            print("     FINGRID_API_KEY=your_key_here")
        return False
    # Check key is set
    content = env_file.read_text()
    if "your_fingrid_api_key_here" in content:
        print("  ⚠  .env exists but FINGRID_API_KEY is still the placeholder")
        return False
    print("  ✔  .env found and API key is set")
    return True


def copy_model_files(model_dir: Path):
    DATA_DIR.mkdir(exist_ok=True)
    print(f"\nCopying model files from: {model_dir.resolve()}")

    all_ok = True
    for fname in REQUIRED_MODEL_FILES:
        src = model_dir / fname
        dst = DATA_DIR / fname
        if not src.exists():
            print(f"  ✗  MISSING (required): {fname}")
            all_ok = False
        else:
            shutil.copy2(src, dst)
            size_kb = dst.stat().st_size // 1024
            print(f"  ✔  {fname:35s} ({size_kb} KB)")

    for fname in OPTIONAL_MODEL_FILES:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, DATA_DIR / fname)
            print(f"  ✔  {fname:35s} (optional)")
        else:
            print(f"  -  {fname:35s} (optional — not found, skipping)")

    return all_ok


def copy_db(db_path: Path):
    DATA_DIR.mkdir(exist_ok=True)
    dst = DATA_DIR / "project_data.db"
    if db_path.exists():
        if dst.resolve() != db_path.resolve():
            shutil.copy2(db_path, dst)
            size_mb = dst.stat().st_size // (1024 * 1024)
            print(f"\n  ✔  Copied project_data.db ({size_mb} MB) → data/")
        else:
            print(f"\n  ✔  project_data.db already in data/ ({db_path.stat().st_size // (1024*1024)} MB)")
        return True
    else:
        print(f"\n  ✗  DB not found at: {db_path}")
        print("     The streamer will create it on first run, but")
        print("     you will need to wait for historical data to load.")
        return False


def validate_data_dir():
    print("\nValidating data/ directory contents:")
    all_ok = True
    for fname in REQUIRED_MODEL_FILES:
        p = DATA_DIR / fname
        if p.exists():
            print(f"  ✔  {fname}")
        else:
            print(f"  ✗  {fname}  ← MISSING")
            all_ok = False

    db = DATA_DIR / "project_data.db"
    if db.exists():
        print(f"  ✔  project_data.db ({db.stat().st_size // (1024*1024)} MB)")
    else:
        print("  ⚠  project_data.db not found — streamer will create it")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Set up CarbonPulse Finland for Docker deployment"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing trained model files (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to project_data.db (default: {DEFAULT_DB_PATH})"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("CarbonPulse Finland — Docker Setup")
    print("=" * 55)

    # 1. Check .env
    print("\n1. Checking .env file...")
    env_ok = check_env()

    # 2. Copy model files
    print("\n2. Copying model files...")
    model_dir = args.model_dir
    if not model_dir.exists():
        # Try common Windows OneDrive path pattern
        onedrive = Path.home() / "OneDrive - University of Oulu and Oamk (1)" / \
                   "Documents" / "RA FCC UBICOMP" / "carbonpulse-finland" / \
                   "model outputs" / "outputs_15min"
        if onedrive.exists():
            print(f"  Found model files at OneDrive path: {onedrive}")
            model_dir = onedrive
        else:
            print(f"  ✗  Model directory not found: {model_dir}")
            print("  Run: python setup.py --model-dir <path to outputs_15min>")
            sys.exit(1)

    models_ok = copy_model_files(model_dir)

    # 3. Copy database
    print("\n3. Setting up database...")
    db_ok = copy_db(args.db_path)

    # 4. Final validation
    print("\n4. Final validation...")
    valid = validate_data_dir()

    # Summary
    print("\n" + "=" * 55)
    if models_ok and valid:
        print("✅ Setup complete! Ready to run:")
        print()
        print("   # Copy .env.example to .env and set your API key:")
        print("   cp .env.example .env")
        print()
        print("   # Start all services:")
        print("   docker compose up --build")
        print()
        print("   # Then open:")
        print("   Dashboard : http://localhost:8501")
        print("   API docs  : http://localhost:8000/docs")
        print("   Forecast  : http://localhost:8000/forecast?steps=96")
    else:
        print("⚠  Setup incomplete — fix the missing files above")
        print("   then run: python setup.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
