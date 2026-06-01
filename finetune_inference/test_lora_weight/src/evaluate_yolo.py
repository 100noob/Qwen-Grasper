import os
import glob
import numpy as np
from PIL import Image
from collections import defaultdict

def yolo_to_absolute(yolo_box, img_width, img_height):
    """Convert YOLO relative coordinates (cx, cy, w, h) to absolute coordinates (xmin, ymin, xmax, ymax)"""
    cx, cy, w, h = yolo_box
    
    abs_cx = cx * img_width
    abs_cy = cy * img_height
    abs_w = w * img_width
    abs_h = h * img_height
    
    xmin = abs_cx - (abs_w / 2)
    ymin = abs_cy - (abs_h / 2)
    xmax = abs_cx + (abs_w / 2)
    ymax = abs_cy + (abs_h / 2)
    
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_width, xmax)
    ymax = min(img_height, ymax)
    
    return [xmin, ymin, xmax, ymax]

def calculate_iou(box1, box2):
    """Calculate the IoU of two bounding boxes with absolute coordinates"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0
    return inter_area / union_area

def load_ground_truth(gt_dir, img_dir):
    """Load image dimensions and ground truth YOLO annotations, and convert to absolute coordinates"""
    gt_boxes = defaultdict(list)
    img_files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    
    for img_path in img_files:
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(gt_dir, f"{base_name}.txt")
        
        if not os.path.exists(txt_path):
            continue
            
        with Image.open(img_path) as img:
            img_width, img_height = img.size
            
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    # To only evaluate bounding box accuracy, ignore original color classes and treat all as class 0
                    class_id = 0
                    yolo_coords = [float(x) for x in parts[1:5]]
                    abs_coords = yolo_to_absolute(yolo_coords, img_width, img_height)
                    gt_boxes[base_name].append({
                        'class_id': class_id,
                        'bbox': abs_coords,
                        'matched': False
                    })
    return gt_boxes

def load_predictions(pred_dir, img_dir):
    """Load model predictions"""
    pred_boxes = []
    txt_files = glob.glob(os.path.join(pred_dir, "*.txt"))
    
    for txt_path in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        img_path_jpg = os.path.join(img_dir, f"{base_name}.jpg")
        img_path_png = os.path.join(img_dir, f"{base_name}.png")
        
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            continue
            
        with Image.open(img_path) as img:
            img_width, img_height = img.size
            
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 6:
                    class_id = int(parts[0])
                    conf = float(parts[1])
                    yolo_coords = [float(x) for x in parts[2:6]]
                    abs_coords = yolo_to_absolute(yolo_coords, img_width, img_height)
                    
                    pred_boxes.append({
                        'image_id': base_name,
                        'class_id': class_id,
                        'confidence': conf,
                        'bbox': abs_coords
                    })
    return pred_boxes

def evaluate(gt_dict, pred_list, iou_thresh=0.5):
    """Calculate AP and mAP"""
    pred_list = sorted(pred_list, key=lambda x: x['confidence'], reverse=True)
    
    gt_classes_count = defaultdict(int)
    for img_id, boxes in gt_dict.items():
        for box in boxes:
            gt_classes_count[box['class_id']] += 1
            
    tp_fp = defaultdict(lambda: {'tp': [], 'fp': []})
    
    for pred in pred_list:
        img_id = pred['image_id']
        pred_class = pred['class_id']
        pred_bbox = pred['bbox']
        
        gt_boxes_in_img = [b for b in gt_dict.get(img_id, []) if b['class_id'] == pred_class]
        
        best_iou = 0
        best_gt_idx = -1
        
        for idx, gt in enumerate(gt_boxes_in_img):
            if gt['matched']:
                continue
            iou = calculate_iou(pred_bbox, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
                
        if best_iou >= iou_thresh:
            gt_boxes_in_img[best_gt_idx]['matched'] = True
            tp_fp[pred_class]['tp'].append(1)
            tp_fp[pred_class]['fp'].append(0)
        else:
            tp_fp[pred_class]['tp'].append(0)
            tp_fp[pred_class]['fp'].append(1)
            
    aps = []
    
    print(f"\n{'-'*45}")
    print(f"{'Class':<15} {'Instances':<15} {'AP@' + str(iou_thresh):<15}")
    print(f"{'-'*45}")
    
    for c in gt_classes_count.keys():
        tps = np.cumsum(tp_fp[c]['tp'])
        fps = np.cumsum(tp_fp[c]['fp'])
        
        recalls = tps / gt_classes_count[c] if gt_classes_count[c] > 0 else np.array([0])
        precisions = tps / (tps + fps + 1e-16)
        
        ap = 0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
            
        aps.append(ap)
        print(f"{str(c):<15} {gt_classes_count[c]:<15} {ap:.4f}")
        
    mAP = np.mean(aps) if aps else 0
    
    total_instances = sum(gt_classes_count.values())
    print(f"{'-'*45}")
    print(f"{'all':<15} {total_instances:<15} {mAP:.4f}")
    print(f"{'-'*45}\n")
    return mAP

def run_evaluation(gt_dir, pred_dir, img_dir, iou_thresh=0.5):
    """Module entry function: Run evaluation pipeline"""
    print("Loading Ground Truth...")
    gt_data = load_ground_truth(gt_dir, img_dir)
    
    print("Loading Predictions...")
    if not os.path.exists(pred_dir):
        print(f"Prediction results folder not found: {pred_dir}")
        print("Please provide YOLO format prediction txt results.")
        return None
        
    pred_data = load_predictions(pred_dir, img_dir)
    
    print(f"Evaluating (IoU Threshold = {iou_thresh})...")
    mAP = evaluate(gt_data, pred_data, iou_thresh)
    return mAP
