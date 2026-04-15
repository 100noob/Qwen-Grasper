import os
import sys
from ultralytics import YOLO
from paths import  DATA_YAML,DATASET,ROOT_DIR

def train():
    
    model = YOLO("yolo11s.pt")
    
    model.train(
        data=DATA_YAML,
        epochs=300,         # 增加到 300 轮，确保充分收敛
        imgsz=320,          # 训练基准设为 320
        batch=32,           # 8G 显存可以轻松跑 batch=32，甚至 64
        device=0,           
        workers=4,          
        # --- 核心增强逻辑 ---
        augment=True,       # 开启数据增强
        multi_scale=False,   # 开启多尺度训练，提高对不同分辨率的鲁棒性
        hsv_s=0.4,          # 适当增加饱和度增强，区分颜色
        hsv_v=0.4,          # 增加亮度增强，应对不同光照
        project="color-cube-train",
        name="exp_320_optimized"
    )





def infer_on_val():
    """使用训练好的模型在验证集(val)上进行推理并保存可视化结果"""
    
    # 加载最佳权重模型
    model = YOLO(f"{ROOT_DIR}/runs/detect/color-cube-train/exp_320_optimized/weights/best.pt")
    
    # 在验证集上进行目标检测推理（不计算mAP等指标）
    model.predict(
        source=f"{DATASET}/images/val",
        conf=0.25,
        save=True
    )
    print("推理完成，结果已保存到 runs/detect/predict 目录")



def eval_on_test():
    """在测试集(test)上评估mAP等指标"""

    # 加载训练好的权重
    model=YOLO(f"{ROOT_DIR}/runs/detect/color-cube-train/exp_320_optimized/weights/best.pt")

    # 在 test 集上评估
    metrics = model.val(
        data=DATA_YAML,   # 必须包含 test 路径
        split="test"        # 关键：指定 test 集
    )

    print("===== Test 集评估完成 =====")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")