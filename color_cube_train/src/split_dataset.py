#给源数据集按tran:80%  val:20%做成训练集文件夹
import os
import random
import shutil

# 原始数据目录
img_src = "source_data/src_images"
lab_src = "source_data/src_labels"

# 目标目录
img_train = "dataset/images/train"
img_val   = "dataset/images/val"
img_test="dataset/images/test"

lab_train = "dataset/labels/train"
lab_val   = "dataset/labels/val"
lab_test="dataset/labels/test"

os.makedirs(img_train, exist_ok=True)
os.makedirs(img_val, exist_ok=True)
os.makedirs(img_test,exist_ok=True)

os.makedirs(lab_train, exist_ok=True)
os.makedirs(lab_val, exist_ok=True)
os.makedirs(lab_test,exist_ok=True)

# 获取所有图片名（不含后缀）
images = [f for f in os.listdir(img_src) if f.endswith(".jpg")]
random.shuffle(images)

# 计算划分点（70% train, 20% val, 10% test）
total = len(images)
split_train=int(total*0.70)
split_val=int(total*0.90)

train_imgs=images[:split_train]
val_imgs=images[split_train:split_val]
test_imgs=images[split_val:]


def move(img_list, img_dst, lab_dst):
    for img in img_list:
        name = os.path.splitext(img)[0]
        shutil.copy(
            os.path.join(img_src, img),
            os.path.join(img_dst, img)
        )
        shutil.copy(
            os.path.join(lab_src, name + ".txt"),
            os.path.join(lab_dst, name + ".txt")
        )

