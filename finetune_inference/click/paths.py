import os

# No matter where it is called, this line locks to the physical location of the current file (paths.py)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Predefined common paths
SRC_IMG = os.path.join(ROOT_DIR, "src_images")
MODEL =os.path.join(ROOT_DIR,"model","best.pt")
JSON_FILE=os.path.join(ROOT_DIR,"destset","results.json")
MY_DIR=os.path.join(ROOT_DIR,"my_dir")
JSONL_FILE=os.path.join(ROOT_DIR,"destset","results.jsonl")
