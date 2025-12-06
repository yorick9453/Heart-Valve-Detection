import os
import glob
import yaml
import math
import argparse
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO

# Tool Function
def center(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def iou(b1, b2):
    x1 = max(b1[2], b2[2])
    y1 = max(b1[3], b2[3])
    x2 = min(b1[4], b2[4])
    y2 = min(b1[5], b2[5])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (b1[4] - b1[2]) * (b1[5] - b1[3])
    area2 = (b2[4] - b2[2]) * (b2[5] - b2[3])
    return inter / (area1 + area2 - inter)


def main(config_path):

    # load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = config["model"]
    image_dir = config["image_dir"]
    final_output = config["final_output"]
    imgsz = config.get("imgsz", 640)
    IoU = config.get("iou", 0.5)
    conf_thres = config.get("conf", 0.0005)
    device = config.get("device", 0)

    os.makedirs(os.path.dirname(final_output), exist_ok=True)

    # load model
    model = YOLO(model_path)

    # load images
    image_files = glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True)
    image_files = [
        f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    raw_data = []

    # raw inference
    for img_path in tqdm(image_files, desc="Running inference", unit="img"):
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # NMS_IoU = 0.5
        results = model.predict(img_path, conf=conf_thres, verbose=False, iou= IoU)

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                raw_data.append(f"{img_name} {cls} {conf:.4f} {x1} {y1} {x2} {y2}")

    # Keep maximum consecutive numbers + If IoU == 0, keep the largest confidence
    data = defaultdict(list)
    for line in raw_data:
        p0 = line.split()[0]        
        patient, imgnum = p0.split("_")
        data[patient].append((int(imgnum), line))

    processed = []

    for patient, entries in data.items():

        entries.sort(key=lambda x: x[0])

        unique_nums = sorted(set(i for i, _ in entries))
        max_seq, cur = [], [unique_nums[0]]

        # Find maximum consecutive numbers
        for prev, curr in zip(unique_nums, unique_nums[1:]):
            if curr == prev + 1:
                cur.append(curr)
            else:
                if len(cur) > len(max_seq):
                    max_seq = cur
                cur = [curr]
        if len(cur) > len(max_seq):
            max_seq = cur

        max_seq_set = set(max_seq)
        selected = [line for num, line in entries if num in max_seq_set]

        # Split by image
        img_group = defaultdict(list)
        for line in selected:
            p = line.split()
            imgnum = int(p[0].split("_")[1])
            cls, conf = int(p[1]), float(p[2])
            x1, y1, x2, y2 = map(int, p[3:7])
            img_group[imgnum].append((cls, conf, x1, y1, x2, y2, p[0]))

        # IoU filter
        for imgnum in sorted(img_group.keys()):
            boxes = img_group[imgnum]
            boxes = sorted(boxes, key=lambda b: b[1], reverse=True)

            kept = []
            for b in boxes:
                ok = True
                for k in kept:
                    if iou(b, k) == 0:
                        ok = False
                        break
                if ok:
                    kept.append(b)

            processed.extend(kept)

    # Write and output result
    with open(final_output, "w") as f:
        for b in processed:
            f.write(f"{b[6]} {b[0]} {b[1]:.4f} {b[2]} {b[3]} {b[4]} {b[5]}\n")

    print(f"Final results：{final_output}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
