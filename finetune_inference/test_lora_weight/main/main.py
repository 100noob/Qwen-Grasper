import time
import os
from openai import OpenAI
import base64
import cv2
import json
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from evaluate_yolo import run_evaluation

# Environment variables cleanup: Prevent local proxies from interfering with NetBird connections
os.environ.pop("ALL_PROXY", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

client = OpenAI(
    api_key="EMPTY",
    base_url="http://100.104.242.193:8000/v1",  # Use NetBird IP
    timeout=3600
)

def encode_image_to_base64(image_path, max_size=1024):
    """Encode image to base64 (required format for Qwen3-VL)"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image, please check if the path is correct or if the file exists: {image_path}")
    h, w = img.shape[:2]

    # 2. Proportional scaling (Core: Reduce image tokens)
    scale = max_size / max(h, w)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. Convert to JPEG binary
    _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 4. Convert to base64
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return img_base64

def draw_bboxes_on_image(image_path, json_result, output_path="annotated_image1.jpg"):
    """
    Draw bounding boxes on the image and annotate with block IDs
    """
    # Read original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image:{image_path}")
    
    img_h, img_w = img.shape[:2]

    # Extract JSON string
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, json_result)
    if match:
        json_result = match.group(1).strip()
    
    # Parse JSON results
    try:
        result = json.loads(json_result)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing failed:{e}")
    
    # Iterate through each color block to draw boxes and IDs
    for block in result.get("blocks", []):
        block_id = block.get("block_id")
        bbox = block.get("bbox")
        
        # Convert normalized coordinates to actual pixel coordinates
        if bbox[0] > 1:
            scale = 1000
        else:
            scale = 1
        x1 = int(bbox[0]/scale * img_w)
        y1 = int(bbox[1]/scale * img_h)
        x2 = int(bbox[2]/scale * img_w)
        y2 = int(bbox[3]/scale * img_h)
        
        # Draw rectangle box (Red, line width 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Calculate ID text position (top-left inside the box)
        text_x = x1 + 5
        text_y = y1 + 35
        font_scale = 3
        
        # Draw text background (Black, semi-transparent)
        text_size = cv2.getTextSize(str(block_id), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        cv2.rectangle(img, (text_x, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        
        # Draw ID text (White)
        cv2.putText(img, str(block_id), (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    
    # Save the annotated image
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to: {output_path}")

def save_yolo_prediction(image_path, json_result, pred_txt_path):
    """
    Convert model JSON results to YOLO format txt prediction files
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    img_h, img_w = img.shape[:2]

    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, json_result)
    if match:
        json_result = match.group(1).strip()
    
    try:
        result = json.loads(json_result)
    except json.JSONDecodeError:
        return
        
    with open(pred_txt_path, 'w') as f:
        for block in result.get("blocks", []):
            bbox = block.get("bbox")
            if bbox[0] > 1:
                scale = 1000.0
            else:
                scale = 1.0
            x1 = bbox[0] / scale
            y1 = bbox[1] / scale
            x2 = bbox[2] / scale
            y2 = bbox[3] / scale
            
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            
            # Here, we need to restore the class mapping based on training settings.
            # Currently, all detected boxes have their class_id uniformly set to 0.
            # Since the model returns grasping order but YOLO test_label uses 0-4 for color categories.
            # Directly comparing grasping order with color classes for mAP will cause serious misalignment!
            # To solely evaluate bounding box accuracy, treating both ground truth and prediction as class 0 (single-class detection) is the only reasonable approach.
            # Thus, it is uniformly set to 0 here.
            class_id = 0
                
            conf = 1.0   # Assume confidence is 1.0
            
            f.write(f"{class_id} {conf} {cx} {cy} {w} {h}\n")

def calculate_metrics(test_dir, output_dir):
    """
    Calculate processing time and other statistical metrics
    This is a placeholder logic, you can add more computation code here
    """
    start_time = time.time()
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_extensions)]
    
    total_images = len(image_files)
    successful_images = 0
    total_time = 0
    
    print(f"\n================ Statistics ================")
    print(f"Found {total_images} images to process")
    
    return total_images, start_time

def main():
    test_dir = "/home/brian/桌面/test_lora_weight/test_set/test_img"
    output_dir = "/home/brian/桌面/test_lora_weight/output_set"
    pred_dir = "/home/brian/桌面/test_lora_weight/pred_set"
    gt_dir = "/home/brian/桌面/test_lora_weight/test_set/test_lable" # Ground truth annotations folder
    
    if not os.path.exists(test_dir):
        print(f"Test directory does not exist: {test_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    # Check if pred_set already has content. If txt files exist, skip inference
    existing_preds = [f for f in os.listdir(pred_dir) if f.endswith('.txt')]
    if existing_preds:
        print(f"\n================ Found Existing Predictions ================")
        print(f"Found {len(existing_preds)} prediction files in {pred_dir}, skipping LLM inference and calculating mAP directly...")
    else:
        # Initialize metrics calculation
        total_images, start_time = calculate_metrics(test_dir, output_dir)
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            print(f"Directory {test_dir} contains no image files")
            return
            
        for filename in image_files:
            image_path = os.path.join(test_dir, filename)
            output_path = os.path.join(output_dir, f"annotated_{filename}")
            print(f"\n=============================================")
            print(f"正Found处理图片: {image_path}")
        
            # Convert image to base64
            try:
                base64_image = encode_image_to_base64(image_path)
            except Exception as e:
                print(f"Failed to process image, skipping: {e}")
                continue
        
            # Prompts
            prompt_text = '''Please strictly analyze the image according to the following requirements:
        1. Count the total number of independent color blocks in the image.
        2. Locate each color block using the bounding box format [x1, y1, x2, y2], where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate, with the origin at the top-left of the image;
        '''
        
            # Construct messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]

            schema = {
                "type": "object",
                "properties": {
                    "total_blocks": {
                        "type": "integer",
                        "description": "Total number of color blocks"
                    },
                    "blocks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "block_id": {
                                    "type": "integer",
                                    "description": "Color block ID (incremental integer starting from 1)"
                                },
                                "bbox": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "description": "Bounding box coordinates [x1, y1, x2, y2]"
                                },
                                "color_description": {
                                    "type": "string",
                                    "description": "Color block description (e.g., red, blue, yellow, etc.)"
                                }
                            },
                            "required": ["block_id", "bbox", "color_description"]
                        }
                    }
                },
                "required": ["total_blocks", "blocks"]
            }

            # Call Large Vision Model
            req_start = time.time()
            try:
                response = client.chat.completions.create(
                    model = "qwen-vl",
                    messages=messages,
                    max_tokens=2048,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "color_blocks_analysis",
                            "schema": schema,
                            "strict": True
                        }
                    },
                    extra_body={
                        "chat_template_kwargs": {"thinking":False}
                    }
                )
                print(f"Response costs: {time.time() - req_start:.2f}s")
                
                # Get LVM output result
                result_text = response.choices[0].message.content
                print(f"Generated text: {result_text}")
                
                # Found图片上标注bbox和编号
                draw_bboxes_on_image(image_path, result_text, output_path=output_path)
                
                # Save as YOLO prediction txt
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                pred_txt_path = os.path.join(pred_dir, txt_filename)
                save_yolo_prediction(image_path, result_text, pred_txt_path)
            
            except Exception as e:
                print(f"Error occurred during LVM inference or image annotation: {e}")
            
        print(f"\n=============================================")
        print(f"All inference tasks completed. Total time cost: {time.time() - start_time:.2f} seconds.")
    
    print(f"\n================ Start Calculating mAP ================")
    run_evaluation(gt_dir=gt_dir, pred_dir=pred_dir, img_dir=test_dir, iou_thresh=0.5)

if __name__ == "__main__":
    main()
