import os

# Lock to the physical location of the current file (paths.py)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to files inside the 'best_ncnn_model' subfolder
MODEL_PARAM = os.path.join(ROOT_DIR, "best_ncnn_model", "model.ncnn.param")
MODEL_BIN = os.path.join(ROOT_DIR, "best_ncnn_model", "model.ncnn.bin")

# Predefined other common paths
DATASET = os.path.join(ROOT_DIR, "dataset")