import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import sys
import os
import json
import argparse
import glob
import csv
import time
from pathlib import Path
from collections import deque, Counter

# ===============================
# PLATFORM DETECTION
# ===============================
import platform
IS_RASPBERRY_PI = (
    platform.machine().startswith("arm") or
    platform.machine().startswith("aarch64") or
    os.path.exists("/proc/device-tree/model")
)

if IS_RASPBERRY_PI:
    device = torch.device("cpu")
    print("Raspberry Pi detected — using CPU")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# ===============================
# CONFIG
# ===============================
CONFIG = {
    "yolo_model":       "best_leaf_detector.pt",
    "classifier_model": "tomato_disease_model.pth",
    "classes_path":     "classes.pth",
    "conf_threshold":   0.60,
    "severity_thresholds": {
        "noise_floor":  0.02,
        "mild_max":     0.12,
        "moderate_max": 0.25,
    },
    "hsv_ranges": {
        "yellow": ([18, 40, 80],  [45, 255, 255]),
        "pale":   ([20, 20, 140], [50, 120, 255]),
        "brown":  ([5, 70, 20],   [25, 255, 180]),
        "green":  ([25, 40, 40],  [90, 255, 255]),
    },
    "clahe_clip_limit": 2.0,
    "clahe_grid_size":  (8, 8),
    "yolo_iou":         0.25,
    "yolo_conf":        0.55,
    "min_leaf_area":    3000,

    # Live feed settings
    # Run inference every N frames. Higher = smoother but slower updates.
    # Pi: 10-15. PC with GPU: 3-5.
    "infer_every_n_frames": 10 if IS_RASPBERRY_PI else 4,

    # Camera resolution (lower = faster on Pi)
    "camera_width":  480 if IS_RASPBERRY_PI else 640,
    "camera_height": 360 if IS_RASPBERRY_PI else 480,

    # Majority-vote smoothing window size
    "smoothing_window": 5,

    # Camera index (0 = default webcam / PiCamera)
    "camera_index": 0,
}

# ===============================
# MODEL LOADING
# ===============================
def load_models():
    for path_key in ["yolo_model", "classifier_model", "classes_path"]:
        path = CONFIG[path_key]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: '{path}'. Check CONFIG paths."
            )

    leaf_detector = YOLO(CONFIG["yolo_model"])

    classes = torch.load(CONFIG["classes_path"], weights_only=False)
    num_classes = len(classes)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    model.load_state_dict(
        torch.load(CONFIG["classifier_model"], map_location=device)
    )
    model = model.to(device)
    model.eval()

    return leaf_detector, model, classes


# ===============================
# TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# DISEASE DATABASE
# ===============================
disease_db = {
    "bacterial_spot":                       ("Avoid wet foliage.",       "Use copper fungicides."),
    "early_blight":                         ("Rotate crops.",            "Use Mancozeb."),
    "late_blight":                          ("Destroy infected plants.", "Use Dimethomorph."),
    "leaf_mold":                            ("Reduce humidity.",         "Use Copper sprays."),
    "septoria_leaf_spot":                   ("Use mulch.",               "Apply Chlorothalonil."),
    "spider_mites_two-spotted_spider_mite": ("Reduce stress.",           "Use Neem oil."),
    "target_spot":                          ("Keep leaves dry.",         "Use fungicides."),
    "tomato_yellow_leaf_curl_virus":        ("Control whiteflies.",      "Remove infected plants."),
    "tomato_mosaic_virus":                  ("Use clean seeds.",         "Destroy infected plants."),
    "healthy":                              ("Maintain care.",           "No treatment needed."),
    "powdery_mildew":                       ("Improve airflow.",         "Use sulfur sprays."),
}

def lookup_disease(disease_raw):
    key = disease_raw.split("___")[-1].lower().replace(" ", "_")
    if key in disease_db:
        return disease_db[key]
    for db_key in disease_db:
        if db_key in key or key in db_key:
            return disease_db[db_key]
    return ("Consult an agricultural expert.", "Consult an agricultural expert.")


# ===============================
# CLAHE NORMALIZATION
# ===============================
def apply_clahe(bgr_img):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CONFIG["clahe_clip_limit"],
        tileGridSize=CONFIG["clahe_grid_size"]
    )
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ===============================
# LEAF MASK
# ===============================
def get_leaf_mask(leaf_img):
    hsv = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2HSV)
    lo, hi = CONFIG["hsv_ranges"]["green"]
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    return clean_mask


# ===============================
# SEVERITY ESTIMATION
# ===============================
def estimate_severity(leaf_img):
    leaf_img = apply_clahe(leaf_img)
    leaf_img = cv2.convertScaleAbs(leaf_img, alpha=1.3, beta=15)
    hsv = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2HSV)
    leaf_mask = get_leaf_mask(leaf_img)
    ranges = CONFIG["hsv_ranges"]
    masks = [
        cv2.inRange(hsv, np.array(ranges[c][0]), np.array(ranges[c][1]))
        for c in ["yellow", "pale", "brown"]
    ]
    disease_mask = masks[0] | masks[1] | masks[2]
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, k3)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, k5)
    disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)
    ratio = np.sum(disease_mask > 0) / (np.sum(leaf_mask > 0) + 1e-6)
    t = CONFIG["severity_thresholds"]
    if ratio < t["noise_floor"]:    return "Mild", 0.0
    elif ratio > t["moderate_max"]: return "Severe", ratio
    elif ratio > t["mild_max"]:     return "Moderate", ratio
    else:                           return "Mild", ratio


# ===============================
# PROCESS ONE LEAF CROP
# ===============================
def process_leaf(leaf_img, model, classes):
    leaf_rgb = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2.resize(leaf_rgb, (224, 224)))
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        confidence, pred = torch.max(probs, 1)
    conf = confidence.item()
    disease = classes[pred.item()]
    if conf < CONFIG["conf_threshold"]:
        return None
    if "healthy" in disease.lower():
        return {"disease": disease, "confidence": conf,
                "severity": "None", "damage_ratio": 0.0}
    severity, ratio = estimate_severity(leaf_img)
    return {"disease": disease, "confidence": conf,
            "severity": severity, "damage_ratio": ratio}


# ===============================
# INFERENCE ON ONE FRAME
# ===============================
def run_inference_on_frame(frame, leaf_detector, model, classes):
    results = leaf_detector(
        frame,
        iou=CONFIG["yolo_iou"],
        conf=CONFIG["yolo_conf"],
        verbose=False
    )
    if len(results[0].boxes) == 0:
        return []
    all_results = []
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        if (x2 - x1) * (y2 - y1) < CONFIG["min_leaf_area"]:
            continue
        leaf = frame[y1:y2, x1:x2]
        result = process_leaf(leaf, model, classes)
        if result:
            result["box"] = (x1, y1, x2, y2)
            all_results.append(result)
    return all_results


# ===============================
# COLOR MAP
# ===============================
color_map = {
    "Severe":   (0, 0, 255),
    "Moderate": (0, 165, 255),
    "Mild":     (0, 255, 255),
    "None":     (0, 255, 0),
}


# ===============================
# DRAW OVERLAY
# ===============================
def draw_overlay(frame, all_results, fps, smoothed_disease):
    h, w = frame.shape[:2]

    # Bounding boxes
    for r in all_results:
        x1, y1, x2, y2 = r["box"]
        color = color_map.get(r["severity"], (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{r['disease'].split('___')[-1]} ({r['severity']})"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty = max(y1 - 5, lh + 4)
        cv2.rectangle(frame, (x1, ty - lh - 4), (x1 + lw + 4, ty), color, -1)
        cv2.putText(frame, label, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Info panel
    panel_h = 70
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    disease_label = smoothed_disease if smoothed_disease else "Scanning..."
    prevention, treatment = lookup_disease(smoothed_disease) if smoothed_disease else ("--", "--")
    cv2.putText(panel, f"Diagnosis: {disease_label}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Prevention: {prevention}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Treatment:  {treatment}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # FPS + leaf count
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w - 90, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Leaves: {len(all_results)}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "[S] Snapshot  [Q] Quit",
                (w - 205, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    return np.vstack([frame, panel])


# ===============================
# OPEN CAMERA (Pi + PC)
# ===============================
def open_camera():
    """Try PiCamera2 on Raspberry Pi, fall back to OpenCV VideoCapture."""
    if IS_RASPBERRY_PI:
        try:
            from picamera2 import Picamera2

            class PiCam2Wrapper:
                def __init__(self):
                    self.cam = Picamera2()
                    cfg = self.cam.create_preview_configuration(
                        main={
                            "size": (CONFIG["camera_width"], CONFIG["camera_height"]),
                            "format": "RGB888"
                        }
                    )
                    self.cam.configure(cfg)
                    self.cam.start()
                    time.sleep(0.5)

                def read(self):
                    frame = self.cam.capture_array()
                    return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                def release(self):
                    self.cam.stop()

            print("Using PiCamera2")
            return PiCam2Wrapper()

        except ImportError:
            print("PiCamera2 not found, falling back to OpenCV VideoCapture")
        except Exception as e:
            print(f"PiCamera2 failed ({e}), falling back to OpenCV VideoCapture")

    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CONFIG['camera_index']}. "
            f"Try a different --camera value."
        )
    print(f"Camera opened (index {CONFIG['camera_index']})")
    return cap


# ===============================
# LIVE FEED
# ===============================
def run_live(leaf_detector, model, classes):
    """
    Real-time leaf disease detection from camera.
    Press S to save snapshot + JSON diagnosis.
    Press Q or ESC to quit.
    """
    cap = open_camera()

    frame_count      = 0
    last_results     = []
    fps              = 0.0
    t_prev           = time.time()
    snapshot_dir     = Path("snapshots")
    snapshot_dir.mkdir(exist_ok=True)
    disease_window   = deque(maxlen=CONFIG["smoothing_window"])
    smoothed_disease = None
    infer_every      = CONFIG["infer_every_n_frames"]

    print("\nLive feed started.")
    print("Press [S] to save snapshot | [Q] or [ESC] to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        frame_count += 1

        # Run inference every N frames
        if frame_count % infer_every == 0:
            last_results = run_inference_on_frame(frame, leaf_detector, model, classes)
            if last_results:
                sev = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
                best = max(last_results,
                           key=lambda r: sev[r["severity"]] + r["confidence"])
                disease_window.append(best["disease"].split("___")[-1])
                smoothed_disease = Counter(disease_window).most_common(1)[0][0]
            else:
                disease_window.clear()
                smoothed_disease = None

        # FPS
        now   = time.time()
        fps   = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now

        # Draw and show
        display_frame = draw_overlay(
            frame.copy(), last_results, fps, smoothed_disease
        )
        cv2.imshow("Tomato Disease Detector  [S=Snapshot  Q=Quit]", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # Snapshot
        if key == ord("s") or key == ord("S"):
            ts       = time.strftime("%Y%m%d_%H%M%S")
            img_path = snapshot_dir / f"snapshot_{ts}.jpg"
            cv2.imwrite(str(img_path), display_frame)
            if last_results:
                sev  = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
                best = max(last_results,
                           key=lambda r: sev[r["severity"]] + r["confidence"])
                prev, treat = lookup_disease(best["disease"])
                diag = {
                    "timestamp":  ts,
                    "disease":    best["disease"],
                    "confidence": round(best["confidence"], 4),
                    "severity":   best["severity"],
                    "damage_pct": round(best["damage_ratio"] * 100, 2),
                    "prevention": prev,
                    "treatment":  treat,
                    "leaf_count": len(last_results),
                }
                json_path = snapshot_dir / f"snapshot_{ts}.json"
                with open(json_path, "w") as f:
                    json.dump(diag, f, indent=2)
                print(f"Snapshot: {img_path}")
                print(f"Diagnosis: {json_path}")
                print(f"  -> {best['disease']} | {best['severity']} | "
                      f"{best['confidence']*100:.1f}%")
            else:
                print(f"Snapshot saved: {img_path} (no leaves detected)")

        # Quit
        elif key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Live feed stopped.")


# ===============================
# AGGREGATE
# ===============================
def aggregate_results(all_results):
    total = len(all_results)
    if total == 0:
        return {}
    disease_counts  = {}
    severity_counts = {"None": 0, "Mild": 0, "Moderate": 0, "Severe": 0}
    total_damage    = 0.0
    for r in all_results:
        d = r["disease"].split("___")[-1]
        disease_counts[d] = disease_counts.get(d, 0) + 1
        severity_counts[r["severity"]] += 1
        total_damage += r["damage_ratio"]
    return {
        "total_leaves":          total,
        "disease_distribution":  disease_counts,
        "severity_distribution": severity_counts,
        "avg_damage_ratio":      total_damage / total,
        "field_health_pct":      (disease_counts.get("healthy", 0) / total) * 100,
    }


# ===============================
# SINGLE IMAGE PIPELINE
# ===============================
def run_pipeline(image_path, leaf_detector, model, classes,
                 display=True, save_output=True):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return {}
    all_results = run_inference_on_frame(image, leaf_detector, model, classes)
    if not all_results:
        print(f"No confident predictions for: {image_path}")
        return {}
    sev  = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    best = max(all_results, key=lambda r: sev[r["severity"]] + r["confidence"])
    prevention, treatment = lookup_disease(best["disease"])
    summary = aggregate_results(all_results)
    output = {
        "image": image_path,
        "best_diagnosis": {
            "disease":      best["disease"],
            "confidence":   round(best["confidence"], 4),
            "severity":     best["severity"],
            "damage_ratio": round(best["damage_ratio"], 4),
            "prevention":   prevention,
            "treatment":    treatment,
        },
        "summary":    summary,
        "all_leaves": [{k: v for k, v in r.items() if k != "box"} for r in all_results],
    }
    print(f"\n===== DIAGNOSIS: {image_path} =====")
    print(f"Disease    : {best['disease']}")
    print(f"Confidence : {best['confidence']*100:.2f}%")
    print(f"Severity   : {best['severity']}")
    print(f"Affected   : {best['damage_ratio']*100:.2f}%")
    print(f"Prevention : {prevention}")
    print(f"Treatment  : {treatment}")
    print(f"\n--- Field Summary ({summary['total_leaves']} leaves) ---")
    print(f"Field Health : {summary['field_health_pct']:.1f}% healthy")
    print(f"Avg Damage   : {summary['avg_damage_ratio']*100:.2f}%")
    print(f"Severity     : {summary['severity_distribution']}")
    print(f"Diseases     : {summary['disease_distribution']}")
    print("=" * 40)

    annotated = image.copy()
    for r in all_results:
        x1, y1, x2, y2 = r["box"]
        color = color_map.get(r["severity"], (200, 200, 200))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{r['disease'].split('___')[-1]} ({r['severity']})"
        cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if save_output:
        out_path = str(Path(image_path).with_suffix("")) + "_annotated.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"Saved: {out_path}")
    if display:
        cv2.imshow("Detected Leaves", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return output


# ===============================
# BATCH
# ===============================
def run_batch(pattern, leaf_detector, model, classes,
              display=False, export_csv=True, export_json=True):
    image_paths = glob.glob(pattern)
    if not image_paths:
        print(f"No images matched: {pattern}")
        return
    print(f"Found {len(image_paths)} image(s)...")
    all_outputs = []
    for path in image_paths:
        result = run_pipeline(path, leaf_detector, model, classes,
                              display=display, save_output=True)
        if result:
            all_outputs.append(result)
    if not all_outputs:
        print("No results to export.")
        return
    if export_json:
        with open("batch_results.json", "w") as f:
            json.dump(all_outputs, f, indent=2)
        print("JSON saved: batch_results.json")
    if export_csv:
        with open("batch_results.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "disease", "confidence", "severity",
                        "damage_%", "field_health_%", "avg_damage_%",
                        "prevention", "treatment"])
            for o in all_outputs:
                bd, sm = o["best_diagnosis"], o.get("summary", {})
                w.writerow([
                    o["image"], bd["disease"],
                    f"{bd['confidence']*100:.2f}", bd["severity"],
                    f"{bd['damage_ratio']*100:.2f}",
                    f"{sm.get('field_health_pct', 0):.1f}",
                    f"{sm.get('avg_damage_ratio', 0)*100:.2f}",
                    bd["prevention"], bd["treatment"],
                ])
        print("CSV saved: batch_results.csv")


# ===============================
# CLI
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Tomato Leaf Disease Detector — Image / Batch / Live Camera"
    )
    parser.add_argument(
        "input", nargs="?",
        help="Image path or glob. Omit to start live camera feed."
    )
    parser.add_argument("--live",        action="store_true",
                        help="Force live camera mode")
    parser.add_argument("--camera",      type=int,   default=None,
                        help="Camera index (default: 0)")
    parser.add_argument("--no-display",  action="store_true",
                        help="Disable imshow (headless mode)")
    parser.add_argument("--json",        action="store_true",
                        help="Print JSON result to stdout")
    parser.add_argument("--batch",       action="store_true",
                        help="Batch mode — input is a glob pattern")
    parser.add_argument("--no-save",     action="store_true",
                        help="Do not save annotated output images")
    parser.add_argument("--iou",         type=float, default=None,
                        help="YOLO NMS IoU threshold (default 0.25)")
    parser.add_argument("--conf",        type=float, default=None,
                        help="YOLO detection confidence (default 0.55)")
    parser.add_argument("--min-area",    type=int,   default=None,
                        help="Min leaf box area in pixels (default 3000)")
    parser.add_argument("--infer-every", type=int,   default=None,
                        help="Live: run inference every N frames (default 4/10)")
    return parser.parse_args()


# ===============================
# ENTRY
# ===============================
if __name__ == "__main__":
    args = parse_args()

    if args.iou         is not None: CONFIG["yolo_iou"]             = args.iou
    if args.conf        is not None: CONFIG["yolo_conf"]            = args.conf
    if args.min_area    is not None: CONFIG["min_leaf_area"]        = args.min_area
    if args.infer_every is not None: CONFIG["infer_every_n_frames"] = args.infer_every
    if args.camera      is not None: CONFIG["camera_index"]         = args.camera

    leaf_detector, model, classes = load_models()

    # Live mode: no input given, or --live flag
    if args.live or args.input is None:
        run_live(leaf_detector, model, classes)

    elif args.batch:
        run_batch(args.input, leaf_detector, model, classes,
                  display=not args.no_display)

    else:
        result = run_pipeline(
            args.input, leaf_detector, model, classes,
            display=not args.no_display,
            save_output=not args.no_save
        )
        if args.json and result:
            print("\n--- JSON Output ---")
            print(json.dumps(result, indent=2))