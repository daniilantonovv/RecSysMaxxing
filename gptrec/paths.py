"""Where datasets and experiment outputs live.

By default they sit next to the package, in the repository. Set the GPTREC_HOME environment variable
before importing `gptrec` to put them elsewhere, e.g. when the code is mounted read-only.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("GPTREC_HOME", Path(__file__).resolve().parents[1]))

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"  # model checkpoints and experiment results
