"""Central project paths.

We keep all paths in a single place (project requirement: no hardcoded paths).
This way, changing the folder layout only requires an edit here instead of in many files.
Paths are computed relative to the location of THIS file, so they work regardless of
the directory from which you run the program.
"""

from pathlib import Path

# __file__ is the path to this file (src/paths.py).
# .resolve() turns it into a full, absolute path.
# .parents[1] goes up two levels: src/paths.py -> src/ -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
GOLDEN_DATASET_PATH = DATA_DIR / "golden_dataset.jsonl"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"

# ChromaDB vector store (local, generated artifact — git-ignored).
CHROMA_DIR = PROJECT_ROOT / "chroma"

# Configuration directory (SLA thresholds, model definitions)
CONFIG_DIR = PROJECT_ROOT / "config"
THRESHOLDS_PATH = CONFIG_DIR / "thresholds.yaml"
MODELS_PATH = CONFIG_DIR / "models.yaml"