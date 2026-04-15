import os
import sys
from ultralytics import YOLO
from paths import DATA_YAML, DATASET, ROOT_DIR

def train():
    model = YOLO("yolo11s.pt")
    
    model.train(
        data=DATA_YAML,
        epochs=300,         
        imgsz=320,          
        batch=32,           
        device=0,           
        workers=4,          
        augment=True,       
        multi_scale=False,   
        hsv_s=0.4,          
        hsv_v=0.4,          
        project="color-cube-train",
        name="exp_320_optimized"
    )

def infer_on_val():
    """Perform inference on the validation set and save visualized results"""
    
    # Load best weights
    model = YOLO(f"{ROOT_DIR}/runs/detect/color-cube-train/exp_320_optimized/weights/best.pt")
    
    # Run object detection inference (metrics like mAP are not calculated here)
    model.predict(
        source=f"{DATASET}/images/val",
        conf=0.25,
        save=True
    )
    print("Inference completed. Results saved to runs/detect/predict directory.")

def eval_on_test():
    """Evaluate metrics like mAP on the test set"""

    # Load trained weights
    model = YOLO(f"{ROOT_DIR}/runs/detect/color-cube-train/exp_320_optimized/weights/best.pt")

    # Evaluate on test split
    metrics = model.val(
        data=DATA_YAML,   
        split="test"        
    )

    print("===== Test Set Evaluation Completed =====")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")