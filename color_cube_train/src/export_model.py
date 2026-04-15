import os
from ultralytics import YOLO
from paths import  DATA_YAML,DATASET,ROOT_DIR

def export_all():
    """导出适用于树莓派的模型格式"""

    # 1. 加载训练好的模型
    model_path = f"{ROOT_DIR}/runs/detect/color-cube-train/exp_320_optimized/weights/best.pt"

    if not os.path.exists(model_path):
        print(" 模型不存在，请先训练")
        return

    model = YOLO(model_path)

    print(" 开始导出模型...")

    # 2. 导出 ONNX（通用）
    model.export(
        format="onnx",
        imgsz=320,  #树莓派480卡，改成320
        opset=12,   # 树莓派兼容性更好
        simplify=True
    )
    print(" ONNX 导出完成")

    # 3. 导出 NCNN（树莓派推荐）
    try:
        model.export(
            format="ncnn",
            imgsz=320
        )
        print(" NCNN 导出完成（推荐用于树莓派）")
    except Exception as e:
        print(" NCNN 导出失败（可能环境不支持）:", e)

    print("所有导出完成")