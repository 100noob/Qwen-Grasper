import os

# No matter where it is called, this line locks to the physical location of the current file (paths.py)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Predefined common paths
DATA_YAML = os.path.join(ROOT_DIR, "data.yaml")
DATASET = os.path.join(ROOT_DIR, "dataset")