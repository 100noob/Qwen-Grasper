import os
import cv2
import ncnn
import numpy as np
import time
from paths import ROOT_DIR,MODEL_PARAM,MODEL_BIN


# Force compatibility configuration to fix Wayland errors on Raspberry Pi 5
os.environ["QT_QPA_PLATFORM"] = "xcb" 

# --- Configuration ---
# MODEL_PARAM = "model.ncnn.param"
# MODEL_BIN = "model.ncnn.bin"
IMG_SIZE = 320 
CONF_THRESHOLD = 0.25  # Slightly lower during testing phase
NMS_THRESHOLD = 0.45
CLASS_NAMES = ["purple-cube", "green-cube", "orange-cube", "pink-cube", "yellow-cube"]

class YOLO11_Detector:
    def __init__(self):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False 
        self.net.opt.num_threads = 4
        self.net.load_param(MODEL_PARAM)
        self.net.load_model(MODEL_BIN)

    def detect(self, frame):
        h, w = frame.shape[:2]
        # Preprocessing
        mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, IMG_SIZE, IMG_SIZE)
        mat_in.substract_mean_normalize([0,0,0], [1/255.0, 1/255.0, 1/255.0])

        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        ret, mat_out = ex.extract("out0")

        out = np.array(mat_out)
        
        # --- Core fix: handle matrix dimension transposition ---
        if len(out.shape) == 2 and out.shape[0] < out.shape[1]:
            out = out.T 
            
        boxes, scores, class_ids = [], [], []

        for i in range(out.shape[0]):
            row = out[i]
            class_scores = row[4:] 
            score = np.max(class_scores)
            
            if score > CONF_THRESHOLD:
                cls_id = int(np.argmax(class_scores))
                if cls_id >= len(CLASS_NAMES):
                    continue

                # Restore coordinates to original image scale
                cx = row[0] * w / IMG_SIZE
                cy = row[1] * h / IMG_SIZE
                bw = row[2] * w / IMG_SIZE
                bh = row[3] * h / IMG_SIZE
                
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                
                boxes.append([x, y, int(bw), int(bh)])
                scores.append(float(score))
                class_ids.append(cls_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
        
        final_results = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_results.append({
                    "box": boxes[i],
                    "class_id": class_ids[i],  # Unified key name as class_id
                    "score": scores[i]
                })
        return final_results

def main():
    detector = YOLO11_Detector()
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret: break

        results = detector.detect(frame)
        
        for res in results:
            x, y, bw, bh = res["box"]
            idx = res["class_id"]
            conf = res["score"]
            
            # Safely get label text
            label_text = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            # Draw label background and text
            cv2.rectangle(frame, (x, y - 20), (x + 120, y), (0, 255, 0), -1)
            cv2.putText(frame, f"{label_text} {conf:.2f}", (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        fps = 1 / (time.time() - t_start)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("YOLO11 RPi5 Real-time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()