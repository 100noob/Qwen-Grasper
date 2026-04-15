import os

# 无论在哪里调用，这行代码都会锁定到当前文件（paths.py）所在的物理位置
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 预定义常用路径
DATA_YAML = os.path.join(ROOT_DIR, "data.yaml")
DATASET = os.path.join(ROOT_DIR, "dataset")