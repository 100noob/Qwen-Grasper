import os

# No matter where it is called, this line locks to the physical location of the current file (paths.py)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


INPUT_JSONL = os.path.join(ROOT_DIR,"JSONL","results.jsonl")
IMAGE_DIR = os.path.join(ROOT_DIR,"src_images")
OUTPUT_JSONL = os.path.join(ROOT_DIR,"JSONL","results_finetune.jsonl")