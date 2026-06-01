import os

# No matter where it is called, this line locks to the physical location of the current file (paths.py)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Predefined common paths
FILE_NAME = os.path.join(ROOT_DIR, "file_name.....")
