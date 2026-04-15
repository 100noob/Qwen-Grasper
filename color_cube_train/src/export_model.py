import os
from ultralytics import YOLO
from paths import DATA_YAML, DATASET, ROOT_DIR

def export_all():
    """Export the model to formats suitable for Raspberry Pi"""

    # 1. Load the trained model
    model_path = f"{ROOT_DIR}/runs/detect/color-cube-train/exp_320_optimized/weights/best.pt"

    if not os.path.exists(model_path):
        print("Model does not exist, please train first.")
        return

    model = YOLO(model_path)

    print("Starting model export...")

    # 2. Export to ONNX (General purpose)
    model.export(
        format="onnx",
        imgsz=320,  # Resolution adjusted to 320 for Raspberry Pi performance
        opset=12,   # Better compatibility for Raspberry Pi
        simplify=True
    )
    print("ONNX export completed.")

    # 3. Export to NCNN (Recommended for Raspberry Pi)
    try:
        model.export(
            format="ncnn",
            imgsz=320
        )
        print("NCNN export completed (Recommended for Raspberry Pi).")
    except Exception as e:
        print(f"NCNN export failed (Environment might not be supported): {e}")

    print("All exports finished.")