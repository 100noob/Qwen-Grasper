import cv2
import os
import shutil
import json
from ultralytics import YOLO
from paths import SRC_IMG, MODEL, JSON_FILE, MY_DIR, JSONL_FILE

# Global variables
detections = []
click_count = 0
click_box_coords = []
is_terminated = False  # Added: Used to determine if the program should exit completely

def mouse_callback(event, x, y, flags, param):
    global click_count, click_box_coords, detections
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Prevent duplicate clicks on the same box
                if any(c[0] == i for c in click_box_coords):
                    return
                
                click_count += 1
                click_box_coords.append((i, (x, y), (x1, y1, x2, y2)))
                # ... (Drawing logic remains unchanged) ...
                cv2.rectangle(param, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(param, f"{click_count}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow("YOLO Detection Result", param)
                break

def yolo_detect_and_mouse_interact(img_path):
    global detections, click_box_coords, is_terminated
    
    model = YOLO(MODEL)
    results = model.predict(img_path, conf=0.25, iou=0.3, save=False)
    detections = results[0].boxes.data.cpu().numpy()
    annotated_img = results[0].plot()

    cv2.namedWindow("YOLO Detection Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Detection Result", 1152, 648)
    cv2.setMouseCallback("YOLO Detection Result", mouse_callback, annotated_img)

    status = "normal" 
    while True:
        cv2.imshow("YOLO Detection Result", annotated_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC: Finish current image, move to next
            break
        if key == ord('q'): # Press Q to exit the program completely without saving the current image
            is_terminated = True
            status = "quit"
            break

    # Sorting logic
    new_order = [idx for idx, _, _ in click_box_coords]
    if new_order:
        detections = detections[new_order]

    cv2.destroyAllWindows()
    return status

def get_last_idx(jsonl_path):
    """Retrieve the maximum idx in the jsonl to ensure continuous numbering"""
    if not os.path.exists(jsonl_path):
        return -1
    last_idx = -1
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                last_idx = max(last_idx, data.get("idx", -1))
            except:
                continue
    return last_idx

if __name__ == "__main__":
    jpg_dir = SRC_IMG
    mv_dir = MY_DIR
    jsonl_file = JSONL_FILE
    
    # 1. Instead of clearing JSONL, get the current maximum index
    idx = get_last_idx(jsonl_file) + 1

    # 2. Iterate through images
    for root, dirs, files in os.walk(jpg_dir):
        if is_terminated: break
        
        for ifile in files:
            if ifile.endswith('.jpg'):
                img_path = os.path.join(root, ifile)

                # Reset status for a single image
                detections = []
                click_count = 0
                click_box_coords = []

                # 3. Get detection status
                status = yolo_detect_and_mouse_interact(img_path)
                
                if is_terminated:
                    print("🛑 Exit command detected, program terminated.")
                    break

                # 4. Only save and move files if exited normally via ESC
                data = {
                    "idx": idx,
                    "jpg": ifile,
                    "detections": detections.tolist()
                }

                with open(jsonl_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

                if not os.path.exists(mv_dir): os.makedirs(mv_dir)
                shutil.move(img_path, os.path.join(mv_dir, ifile))
                
                print(f"✅ Processed: {ifile} (idx: {idx})")
                idx += 1

    # 5. Generate final JSON
    if os.path.exists(jsonl_file):
        from paths import JSON_FILE
        results = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                results.append(json.loads(line))
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print("\n🎉 JSON has been updated.")
