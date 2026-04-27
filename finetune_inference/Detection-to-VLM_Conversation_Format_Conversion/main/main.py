import json
import os
from PIL import Image
from paths import INPUT_JSONL, IMAGE_DIR, OUTPUT_JSONL



def normalize_box(box, w, h):
    x1, y1, x2, y2 = box
    return [
        round(x1 / w, 4),
        round(y1 / h, 4),
        round(x2 / w, 4),
        round(y2 / h, 4)
    ]


def sort_boxes(dets):

    clean_dets = []

    for i, det in enumerate(dets):

        if not isinstance(det, (list, tuple)) or len(det) < 4:
            print("BAD DETECT:", det)
            print("PREVIOUS DETECT:", dets[i - 1] if i > 0 else None)
            continue

        clean_dets.append(det)

    return sorted(clean_dets, key=lambda x: x[1])


def process():
    with open(INPUT_JSONL, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:

        for i, line in enumerate(f_in):
            data = json.loads(line)

            img_name = data["jpg"]
            img_path = os.path.join(IMAGE_DIR, img_name)

            # Get image dimensions
            with Image.open(img_path) as img:
                w, h = img.size

            detections = data["detections"]

            # Sort (strategy can be modified)
            detections = sort_boxes(detections)

            # Convert bbox
            boxes = [
                normalize_box(det[:4], w, h)
                for det in detections
            ]

            new_item = {
                "idx": f"scene_{i:03d}",
                "image": img_name,
                "conversations": [
                    {
                        "from": "user",
                        "value": (
                            "1. Count the number of independent color blocks in the image.\n"
                            "2. Localize each color block using bounding boxes in the format [x1, y1, x2, y2], "
                            "where coordinates are normalized (0–1) relative to image size, origin at top-left.\n"
                            "3. Order the blocks from easiest to hardest to grasp."
                        )
                    },
                    {
                        "from": "assistant",
                        "value": boxes
                    }
                ]
            }

            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    process()
