import os

# 锁定当前文件所在目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 指向子文件夹 best_ncnn_model 里的文件
MODEL_PARAM = os.path.join(ROOT_DIR, "best_ncnn_model", "model.ncnn.param")
MODEL_BIN = os.path.join(ROOT_DIR, "best_ncnn_model", "model.ncnn.bin")

# 预定义其他路径
DATASET = os.path.join(ROOT_DIR, "dataset")