import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import json
import argparse
import glob
import csv
import time
import threading
from pathlib import Path
from collections import deque, Counter

# ── ESP32 Serial ─────────────────────────────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("pyserial not installed. Run: pip install pyserial")

# ===============================
# LOGGING
# ===============================
import logging
import platform

_log_path = "tomcla.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_path),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("tomcla")

# ===============================
# PLATFORM DETECTION
# ===============================
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

    # ── ESP32 Serial Settings ─────────────────────────────────────────────
    # Set esp32_port to None to auto-detect, or set manually e.g. "COM5"
    "esp32_port":     None,
    "esp32_baudrate": 115200,
    "esp32_enabled":  True,

    # How many prevention steps to send to LCD (16x2 fits 2 lines of 16 chars)
    # Steps are split and scrolled on the ESP32 side
    "lcd_steps_to_send": 5,
}

# ===============================
# ESP32 SERIAL MANAGER
# ===============================
class ESP32Serial:
    """
    Manages USB serial connection to ESP32.
    Sends disease name + prevention steps as simple text lines.
    ESP32 displays them on the 16x2 LCD.
    """
    def __init__(self):
        self.ser  = None
        self.port = None
        self._lock = threading.Lock()

    def auto_detect_port(self):
        """Find ESP32 COM port automatically."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            if any(k in desc or k in hwid for k in
                   ["cp210", "ch340", "ch341", "esp32", "usb serial", "uart"]):
                return p.device
        # Fallback: return first available port
        if ports:
            return ports[0].device
        return None

    def connect(self, port=None, baudrate=115200):
        """Connect to ESP32. Auto-detects port if not specified."""
        if not SERIAL_AVAILABLE:
            log.warning("pyserial not available — ESP32 disabled")
            return False
        try:
            port = port or CONFIG.get("esp32_port") or self.auto_detect_port()
            if port is None:
                log.warning("No ESP32 port found. Check USB connection.")
                return False
            self.ser  = serial.Serial(port, baudrate, timeout=2)
            self.port = port
            time.sleep(2)  # Wait for ESP32 to reset after serial connect
            log.info(f"ESP32 connected on {port} at {baudrate} baud")
            return True
        except Exception as e:
            log.error(f"ESP32 connection failed: {e}")
            return False

    def send(self, text):
        """Send a line of text to ESP32 (non-blocking)."""
        if self.ser and self.ser.is_open:
            try:
                with self._lock:
                    self.ser.write((text.strip() + "\n").encode("utf-8"))
                    self.ser.flush()
            except Exception as e:
                log.error(f"ESP32 send error: {e}")

    def send_diagnosis(self, disease_name, severity, prevention_text):
        """
        Format and send diagnosis to ESP32 LCD.
        Protocol (each line ends with \n):
          DISEASE:<name>
          SEVERITY:<level>
          STEP:<step text>   (repeated for each step)
          END
        """
        if not self.ser or not self.ser.is_open:
            return

        # Clean disease name for LCD (max 16 chars)
        name = disease_name.split("___")[-1].replace("_", " ").title()[:16]
        sev  = severity[:8]

        lines = []
        lines.append(f"DISEASE:{name}")
        lines.append(f"SEVERITY:{sev}")

        # Parse prevention steps from the multiline string
        steps = []
        for line in prevention_text.splitlines():
            line = line.strip()
            # Match lines like "Step 1: ..." or "  Step 2: ..."
            if line.lower().startswith("step"):
                # Extract just the instruction part after the colon
                parts = line.split(":", 1)
                if len(parts) > 1:
                    step_text = parts[1].strip()
                    steps.append(step_text)

        for i, step in enumerate(steps[:CONFIG["lcd_steps_to_send"]], 1):
            # Truncate each step to 32 chars (ESP32 will scroll on LCD)
            lines.append(f"STEP{i}:{step[:60]}")

        lines.append("END")

        for line in lines:
            self.send(line)
            time.sleep(0.05)  # Small delay between lines

        log.info(f"Sent to ESP32: {name} | {sev} | {len(steps)} steps")

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            log.info("ESP32 disconnected")

    @property
    def connected(self):
        return self.ser is not None and self.ser.is_open


# Global ESP32 instance
esp32 = ESP32Serial()


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

    # torch.compile gives 15-30% speedup on Pi 5 and modern CPUs
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile applied")
    except Exception:
        pass  # torch.compile not available on older PyTorch — skip silently

    # Warmup inference: first run is always slow due to JIT kernel compilation.
    # Running a dummy input now makes the first real image fast.
    print("Running warmup inference...")
    _dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.inference_mode():
        model(_dummy)
    print("Warmup done. Ready.")

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
# Each entry: (prevention_steps, treatment_steps)
# Written in simple language for uneducated farmers.
# ===============================
disease_db = {
    "bacterial_spot": (
        """PREVENTION (Stop it before it spreads):
  Step 1: Do NOT water the leaves. Water only at the base of the plant.
  Step 2: Remove and burn any yellow or spotted leaves immediately.
  Step 3: Do not work in the field when plants are wet (after rain or morning dew).
  Step 4: Keep distance between plants so air can flow between them.
  Step 5: Do not reuse seeds from infected plants next season.""",

        """TREATMENT (How to cure infected plants):
  Step 1: Mix copper fungicide powder with water (follow packet instructions).
  Step 2: Spray the mixture on ALL leaves — top and bottom surfaces.
  Step 3: Spray every 7 days for 3 weeks.
  Step 4: After spraying, wash your hands and clothes.
  Step 5: If plant is more than 50% affected, remove and burn the whole plant."""
    ),

    "early_blight": (
        """PREVENTION (Stop it before it spreads):
  Step 1: Do not plant tomatoes in the same spot every year. Change the location.
  Step 2: Remove old dead leaves from the ground — they carry the disease.
  Step 3: Water only at the roots, never on the leaves.
  Step 4: Keep leaves dry — water early morning so leaves dry by noon.
  Step 5: Space plants at least 50cm apart for good air flow.""",

        """TREATMENT (How to cure infected plants):
  Step 1: Pick off and burn all yellow/brown leaves with dark rings on them.
  Step 2: Buy Mancozeb powder from an agri shop.
  Step 3: Mix 2 grams of Mancozeb per 1 litre of water.
  Step 4: Spray on all leaves every 7-10 days.
  Step 5: Continue spraying for at least 3 sprays even if plant looks better."""
    ),

    "late_blight": (
        """PREVENTION (Stop it before it spreads):
  Step 1: This disease spreads very fast in cool, wet weather — watch plants daily.
  Step 2: Avoid planting tomatoes near potato plants (same disease can spread).
  Step 3: Do NOT water in the evening — wet leaves at night causes this disease.
  Step 4: Remove and burn any leaves with dark brown/black patches immediately.
  Step 5: Never compost infected plants — always burn them.""",

        """TREATMENT (How to cure infected plants):
  Step 1: Act immediately — this disease can kill the whole plant in 3-5 days.
  Step 2: Remove and burn ALL infected leaves and stems right away.
  Step 3: Buy Dimethomorph or Metalaxyl fungicide from agri shop.
  Step 4: Mix and spray as per packet instructions on all remaining leaves.
  Step 5: Spray every 5 days for 3 rounds.
  Step 6: If more than half the plant is infected, remove the entire plant and burn it."""
    ),

    "leaf_mold": (
        """PREVENTION (Stop it before it spreads):
  Step 1: This disease grows in humid, crowded conditions — give plants more space.
  Step 2: Open greenhouse vents or windows to allow fresh air circulation.
  Step 3: Avoid overhead watering — water only at the soil level.
  Step 4: Remove lower leaves that touch the soil.
  Step 5: Do not plant tomatoes in the same spot two years in a row.""",

        """TREATMENT (How to cure infected plants):
  Step 1: Pick off all leaves showing yellow patches on top or brown dust below.
  Step 2: Buy Copper Oxychloride spray from agri shop.
  Step 3: Mix as per packet and spray under and over every leaf.
  Step 4: Spray once every 7 days for 3 weeks.
  Step 5: Reduce watering and improve air flow around plants immediately."""
    ),

    "septoria_leaf_spot": (
        """PREVENTION (Stop it before it spreads):
  Step 1: Put dry straw or leaves on the soil around the plant (called mulching).
     This stops soil water from splashing onto leaves and spreading the disease.
  Step 2: Do not touch plants when they are wet.
  Step 3: Remove infected lower leaves as soon as you see them.
  Step 4: Rotate crops — do not grow tomatoes in same soil every year.
  Step 5: Clean and wash your farm tools regularly.""",

        """TREATMENT (How to cure infected plants):
  Step 1: Remove all leaves with small white/grey round spots with dark borders.
  Step 2: Buy Chlorothalonil fungicide from agri shop.
  Step 3: Mix as per packet and spray all leaves (top and bottom).
  Step 4: Spray every 7-10 days for at least 3 rounds.
  Step 5: Do not let water touch leaves while spraying or watering."""
    ),

    "spider_mites_two-spotted_spider_mite": (
        """PREVENTION (Stop it before it spreads):
  Step 1: Spider mites grow in hot, dry weather — water plants regularly.
  Step 2: Spray plain water on leaves during hot days to reduce mites.
  Step 3: Remove heavily infested leaves immediately.
  Step 4: Avoid over-using pesticides — they kill natural enemies of mites.
  Step 5: Keep the area around plants weed-free.""",

        """TREATMENT (How to cure infected plants):
  Step 1: Look for tiny spider webs under leaves — that confirms mites.
  Step 2: Buy Neem oil from agri shop (safe and natural).
  Step 3: Mix 5ml Neem oil + 1ml liquid soap + 1 litre water.
  Step 4: Spray under every leaf every 5 days for 3 rounds.
  Step 5: If Neem oil does not work after 3 sprays, buy Abamectin from agri shop."""
    ),

    "target_spot": (
        """PREVENTION (Stop it before it spreads):
  Step 1: Avoid wetting the leaves when watering.
  Step 2: Trim lower leaves that touch the soil.
  Step 3: Keep plants properly spaced for air flow.
  Step 4: Remove fallen leaves and debris from the ground.
  Step 5: Do not plant tomatoes in the same bed every season.""",

        """TREATMENT (How to cure infected plants):
  Step 1: Remove all leaves with brown spots that look like a target/bullseye.
  Step 2: Buy any broad-spectrum fungicide (e.g. Mancozeb or Tebuconazole).
  Step 3: Mix as per packet instructions.
  Step 4: Spray on all leaves every 7 days for 3 rounds.
  Step 5: If disease continues spreading, consult your local agri officer."""
    ),

    "tomato_yellow_leaf_curl_virus": (
        """PREVENTION (Stop it before it spreads):
  Step 1: This disease is spread by tiny white insects called whiteflies.
     Install yellow sticky traps near plants to catch whiteflies.
  Step 2: Cover young plants with fine nets to block whiteflies.
  Step 3: Remove and burn infected plants immediately — this virus has NO cure.
  Step 4: Do not bring plants from unknown sources into your farm.
  Step 5: Plant resistant tomato varieties if available in your area.""",

        """TREATMENT (How to manage infected plants):
  Step 1: There is NO chemical cure for this virus.
  Step 2: Remove and burn infected plants IMMEDIATELY to stop spread.
  Step 3: Spray Imidacloprid insecticide to kill whiteflies on healthy plants.
  Step 4: Mix as per packet — spray every 7 days on healthy plants nearby.
  Step 5: After removing infected plants, wash hands and tools before touching others."""
    ),

    "tomato_mosaic_virus": (
        """PREVENTION (Stop it before it spreads):
  Step 1: Always buy certified, disease-free seeds from a trusted shop.
  Step 2: Do not smoke near plants — tobacco carries this virus.
  Step 3: Wash hands with soap before touching plants.
  Step 4: This virus spreads by touching — do not touch healthy plants after infected ones.
  Step 5: Remove and burn infected plants right away.""",

        """TREATMENT (How to manage infected plants):
  Step 1: There is NO chemical cure for this virus.
  Step 2: Pull out and burn all infected plants immediately.
  Step 3: Clean and disinfect all tools used on infected plants with bleach water.
  Step 4: Spray healthy plants with insecticide to kill insects that spread the virus.
  Step 5: Next season, use virus-resistant tomato seed varieties."""
    ),

    "healthy": (
        """PREVENTION (Keep your plant healthy):
  Step 1: Water regularly at the soil level — not on leaves.
  Step 2: Add compost or fertilizer once every 2-3 weeks.
  Step 3: Check plants every day for early signs of disease.
  Step 4: Keep weeds away from the base of plants.
  Step 5: Ensure plants get at least 6-8 hours of sunlight daily.""",

        """TREATMENT: Your plant is HEALTHY — No treatment needed!
  Keep doing what you are doing. Continue regular care:
  - Water at base, not on leaves.
  - Remove any yellowing leaves early.
  - Check under leaves for pests weekly."""
    ),

    "powdery_mildew": (
        """PREVENTION (Stop it before it spreads):
  Step 1: Plant tomatoes with enough space between them — at least 50cm apart.
  Step 2: Avoid shaded, damp areas for planting.
  Step 3: Water in the morning so leaves stay dry by afternoon.
  Step 4: Remove infected leaves the moment you see white powder on them.
  Step 5: Do not over-fertilize with nitrogen — it makes plants more vulnerable.""",

        """TREATMENT (How to cure infected plants):
  Step 1: Mix 1 tablespoon baking soda + 1 teaspoon liquid soap in 1 litre water.
     Spray this on white powdery areas. It is safe and cheap.
  Step 2: OR buy Sulfur-based fungicide from agri shop.
  Step 3: Spray every 7 days on all leaves (top and bottom).
  Step 4: Remove and burn heavily infected leaves.
  Step 5: Repeat for at least 3 rounds even if the white powder disappears."""
    ),
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
# Created once globally — not per-leaf (performance fix)
# ===============================
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(bgr_img):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
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
    with torch.inference_mode():
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
    # Resize large images before YOLO — phone photos can be 4K+
    # This gives massive speedup on Pi with no meaningful accuracy loss
    h, w = frame.shape[:2]
    max_side = 1280
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

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
                _snap_summary = aggregate_results(last_results)
                diag = {
                    "timestamp":     ts,
                    "disease":       best["disease"],
                    "confidence":    round(best["confidence"], 4),
                    "severity":      best["severity"],
                    "damage_pct":    round(best["damage_ratio"] * 100, 2),
                    "prevention":    prev,
                    "treatment":     treat,
                    "leaf_count":    len(last_results),
                    "field_health_pct":   round(_snap_summary.get("field_health_pct", 0), 1),
                    "avg_damage_pct":     round(_snap_summary.get("avg_damage_ratio", 0) * 100, 1),
                    "disease_distribution": _snap_summary.get("disease_distribution", {}),
                    "severity_distribution": _snap_summary.get("severity_distribution", {}),
                }
                json_path = snapshot_dir / f"snapshot_{ts}.json"
                with open(json_path, "w") as f:
                    json.dump(diag, f, indent=2)
                print(f"Snapshot: {img_path}")
                print(f"Diagnosis: {json_path}")
                print(f"  -> {best['disease']} | {best['severity']} | "
                      f"{best['confidence']*100:.1f}%")
                # Send to ESP32 LCD on snapshot
                if esp32.connected:
                    _prev, _ = lookup_disease(best["disease"])
                    esp32.send_diagnosis(
                        disease_name=best["disease"],
                        severity=best["severity"],
                        prevention_text=_prev,
                    )
            else:
                print(f"Snapshot saved: {img_path} (no leaves detected)")

        # Quit
        elif key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    esp32.disconnect()
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
        log.error(f"Could not read image: {image_path}")
        return {}
    all_results = run_inference_on_frame(image, leaf_detector, model, classes)
    if not all_results:
        log.warning(f"No confident predictions for: {image_path}")
        return {}
    sev  = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    best = max(all_results, key=lambda r: sev[r["severity"]] + r["confidence"])
    prevention, treatment = lookup_disease(best["disease"])
    summary = aggregate_results(all_results)

    # Collect unique diseases — keep WORST severity per disease, not first seen
    sev_rank = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    unique_diseases = {}
    for r in all_results:
        key = r["disease"].split("___")[-1]
        if key not in unique_diseases:
            prev, treat = lookup_disease(r["disease"])
            unique_diseases[key] = {
                "disease":    r["disease"],
                "confidence": round(r["confidence"], 4),
                "severity":   r["severity"],
                "prevention": prev,
                "treatment":  treat,
            }
        else:
            # Update to worst severity seen for this disease
            existing_rank = sev_rank[unique_diseases[key]["severity"]]
            new_rank      = sev_rank[r["severity"]]
            if new_rank > existing_rank:
                unique_diseases[key]["severity"]   = r["severity"]
                unique_diseases[key]["confidence"] = round(r["confidence"], 4)

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
        "all_diseases":  list(unique_diseases.values()),
        "summary":       summary,
        "all_leaves":    [{k: v for k, v in r.items() if k != "box"} for r in all_results],
    }
    print(f"\n===== DIAGNOSIS: {image_path} =====")
    print(f"Primary Disease : {best['disease']}")
    print(f"Confidence      : {best['confidence']*100:.2f}%")
    print(f"Severity        : {best['severity']}")
    print(f"Affected        : {best['damage_ratio']*100:.2f}%")
    print(f"\n--- Treatment Plan ({len(unique_diseases)} disease(s) detected) ---")
    for i, (dkey, info) in enumerate(unique_diseases.items(), 1):
        print(f"\n{'='*50}")
        print(f"  [{i}] {dkey.upper()}")
        print(f"      Severity   : {info['severity']}")
        print(f"      Confidence : {info['confidence']*100:.1f}%")
        print(f"\n{info['prevention']}")
        print(f"\n{info['treatment']}")
    print(f"{'='*50}")
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

    # Send primary disease to ESP32 LCD
    if esp32.connected:
        esp32.send_diagnosis(
            disease_name=best["disease"],
            severity=best["severity"],
            prevention_text=prevention,
        )
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
        _csv_ts = time.strftime("%Y%m%d_%H%M%S")
        _csv_name = f"batch_results_{_csv_ts}.csv"
        with open(_csv_name, "w", newline="") as f:
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
        print(f"CSV saved: {_csv_name}")


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
    parser.add_argument("--esp32-port",  type=str,   default=None,
                        help="ESP32 COM port (e.g. COM5 or /dev/ttyUSB0). Auto-detects if not set.")
    parser.add_argument("--no-esp32",    action="store_true",
                        help="Disable ESP32 LCD output entirely")
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
    if args.esp32_port  is not None: CONFIG["esp32_port"]           = args.esp32_port
    if args.no_esp32:                CONFIG["esp32_enabled"]        = False

    leaf_detector, model, classes = load_models()

    # Connect ESP32 if enabled
    if CONFIG.get("esp32_enabled") and SERIAL_AVAILABLE:
        connected = esp32.connect(
            port=CONFIG.get("esp32_port"),
            baudrate=CONFIG.get("esp32_baudrate", 115200)
        )
        if connected:
            esp32.send("READY:Tomato Doctor")
        else:
            print("ESP32 not connected — LCD display disabled. Continuing without it.")

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