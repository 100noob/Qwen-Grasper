import time
from openai import OpenAI
import base64
import cv2
import json
import re
import os
import sys

# Import paths from paths.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import INPUT_DIR, OUTPUT_DIR

# Environment variable cleanup: prevent local proxies from interfering with NetBird connections
os.environ.pop("ALL_PROXY", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# --- Modification point 1: API access address ---
# Please replace "YOUR_API_KEY" with your actual API key.
# Please replace "YOUR_SERVER_IP" with your actual server IP (e.g. your NetBird IP or local IP).
client = OpenAI(
    api_key="YOUR_API_KEY", 
    base_url="http://YOUR_SERVER_IP:8000/v1",  
    timeout=3600
)

def encode_image_to_base64(image_path, max_size=1024):
    """Convert local image to base64 encoding and send to server"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(img_encoded).decode('utf-8')

def draw_bboxes_on_image(image_path, json_result, output_path):
    """Draw bounding boxes on local image and save to local disk"""
    img = cv2.imread(image_path)
    if img is None: return
    img_h, img_w = img.shape[:2]
    
    # Extract JSON content
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, json_result)
    if match:
        json_result = match.group(1).strip()
    
    try:
        result = json.loads(json_result)
        for block in result.get("blocks", []):
            block_id = block.get("block_id")
            bbox = block.get("bbox")
            # Compatible with normalized coordinates (0-1) and pixel coordinates (0-1000)
            scale = 1000 if any(v > 1 for v in bbox) else 1
            x1, y1 = int(bbox[0]/scale * img_w), int(bbox[1]/scale * img_h)
            x2, y2 = int(bbox[2]/scale * img_w), int(bbox[3]/scale * img_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, str(block_id), (x1 + 5, y1 + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(output_path, img)
    except Exception as e:
        print(f"Local annotation failed: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    
    print(f"Number of local images: {len(image_files)}, requesting inference via API...")

    for file_name in image_files:
        image_path = os.path.join(INPUT_DIR, file_name)
        print(f"\nProcessing local image: {file_name}")
        
        base64_image = encode_image_to_base64(image_path)
        if base64_image is None:
            continue
        
        prompt_text = "Please strictly follow the requirements to analyze the image: 1. Count the number of all independent color blocks 2. Use bounding box format [x1, y1, x2, y2] for positioning."
        
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "text", "text": prompt_text}
        ]}]

        schema = {
            "type": "object",
            "properties": {
                "total_blocks": {"type": "integer"},
                "blocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "block_id": {"type": "integer"},
                            "bbox": {"type": "array", "items": {"type": "number"}},
                            "color_description": {"type": "string"}
                        },
                        "required": ["block_id", "bbox", "color_description"]
                    }
                }
            },
            "required": ["total_blocks", "blocks"]
        }
        
        try:
            start = time.time()
            # --- Modification point 3: Model Name ---
            # Keep it exactly consistent with the --model parameter in the server startup command
            response = client.chat.completions.create(
                model="qwen-vl",  # Must be exactly consistent with "id" returned by server
                messages=messages,
                max_tokens=2048,
                response_format={"type": "json_schema", "json_schema": {"name": "analysis", "schema": schema, "strict": True}},
                extra_body={"chat_template_kwargs": {"thinking": False}}
            )
            
            result_text = response.choices[0].message.content
            print(f"Remote inference time: {time.time() - start:.2f}s")
            
            # Save to local outputs_results folder
            output_path = os.path.join(OUTPUT_DIR, f"result_{file_name}")
            draw_bboxes_on_image(image_path, result_text, output_path)
            print(f"Local result generated: {output_path}")
            
        except Exception as e:
            print(f"Inference error ({file_name}): {repr(e)}")

if __name__ == "__main__":
    main()
