import os

# No matter where it is called, this line locks to the physical location of the current file (paths.py)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Predefined common paths
MODEL_NAME = "/home/user_name/models/Qwen3.5-9B"  # you can select you model path
JSONL_PATH = os.path.join(ROOT_DIR,"imgs_and_json","jsonl","results_finetune.jsonl")
IMAGE_DIR = os.path.join(ROOT_DIR,"imgs_and_json","src_images")


#export model path
EXPORT_WEIGHT=os.path.join(ROOT_DIR,"export_weight")