#!/usr/bin/env python3
"""
Lab 9 — Real-Time Anomaly Detection (no email)
Anomaly = presence of any class in ANOMALY_SET (default: person, laptop).
Logs to detections.log (human-readable) + anomaly_log.csv (for analysis).
Saves a snapshot image per anomaly.
Runs until you close the window or press Ctrl+C.
"""

import os, sys, time, csv, argparse, logging
from logging.handlers import RotatingFileHandler

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, saveImage


# ================== Paths (mounted so they persist) ==================
# Container path that maps to host: /home/jetson1/jetson-inference/data/assignment
BASE_DIR     = "/jetson-inference/data/nandini"
LOG_PATH     = os.path.join(BASE_DIR, "detections.log")
CSV_PATH     = os.path.join(BASE_DIR, "anomaly_log.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "anomaly_images")


# ================== Rules ==================
# Any of these classes present -> anomaly
ANOMALY_SET = {"chair", "cellphone"}

# Optional cap on number of persons (None disables the rule)
MAX_PERSONS = None  # e.g., set to 3 to flag >3 people


# ================== Utilities ==================
def normalize_uri(s: str) -> str:
    """Convert bare /dev/videoN to v4l2:///dev/videoN for USB webcams."""
    if s and s.startswith("/dev/video"):
        return "v4l2://" + s
    return s

def ensure_dirs():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def setup_logging():
    handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler, logging.StreamHandler(sys.stdout)],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Logging to %s", LOG_PATH)

def setup_csv():
    new_file = not os.path.exists(CSV_PATH)
    f = open(CSV_PATH, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["timestamp", "reason", "labels", "counts", "snapshot"])
    return f, w


# ================== CLI ==================
parser = argparse.ArgumentParser(
    description="Lab 9: Real-Time Anomaly Detection (flags person/laptop).",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        "Examples:\n"
        "  python3 lab9_assignment.py /dev/video0 display://0 -- "
        "--input-codec=mjpeg --input-width=1280 --input-height=720\n"
        "  python3 lab9_assignment.py /dev/video0 output.mp4    -- "
        "--input-codec=yuv   --input-width=640  --input-height=480\n"
    ),
)
parser.add_argument("input",  nargs="?", default="/dev/video0", help="input (e.g., /dev/video0, v4l2:///..., csi://0)")
parser.add_argument("output", nargs="?", default="display://0",  help="output (display://0, output.mp4)")
parser.add_argument("--network",   default="ssd-mobilenet-v2")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--overlay",   default="box,labels,conf", help="box,labels,conf,none")
args, _ = parser.parse_known_args()  # extra video args after '--' are forwarded via sys.argv


# ================== Init ==================
ensure_dirs()
setup_logging()
csv_file, csv_writer = setup_csv()

inp_uri = normalize_uri(args.input)
out_uri = args.output

net     = detectNet(args.network, sys.argv, args.threshold)
camera  = videoSource(inp_uri,  argv=sys.argv)   # forwards extra video args after '--'
display = videoOutput(out_uri, argv=sys.argv)

print("------------------------------------------------")
print(f"Input  : {inp_uri}")
print(f"Output : {out_uri}")
print("Anomaly rules:")
print(f"  - Presence of any: {sorted(ANOMALY_SET)}")
if MAX_PERSONS is not None:
    print(f"  - person count > {MAX_PERSONS}")
print(f"Log file : {LOG_PATH}")
print(f"CSV file : {CSV_PATH}")
print(f"Snapshots: {SNAPSHOT_DIR}")
print("Starting detection... Ctrl+C to stop.")
print("------------------------------------------------")

logging.info("Input=%s Output=%s Anomalies=%s MaxPersons=%s",
             inp_uri, out_uri, sorted(ANOMALY_SET), MAX_PERSONS)

anomaly_count = 0
last_reason   = "—"

try:
    while True:
        img = camera.Capture()
        if img is None:
            if not camera.IsStreaming():
                print("[EXIT] Input EOS.")
                logging.info("Exit: input EOS")
                break
            continue

        dets = net.Detect(img, overlay=args.overlay)

        # Collect per-frame labels & counts
        labels = [net.GetClassDesc(d.ClassID).lower() for d in dets]
        counts = {}
        for L in labels:
            counts[L] = counts.get(L, 0) + 1

        present = set(labels)

        # ---------- Anomaly logic ----------
        reasons = []
        # Rule 1: forbidden/present classes
        bad = sorted(list(present.intersection(ANOMALY_SET)))
        if bad:
            reasons.append("Detected: " + ", ".join(bad))

        # Rule 2: person count cap
        if MAX_PERSONS is not None and counts.get("person", 0) > MAX_PERSONS:
            reasons.append(f"person_count>{MAX_PERSONS} (={counts.get('person',0)})")

        is_anomaly = len(reasons) > 0
        reason_txt = " | ".join(reasons) if reasons else "No anomaly classes present"
        # -----------------------------------

        # Print/log each detection
        for d in dets:
            label = net.GetClassDesc(d.ClassID).lower()
            conf  = d.Confidence * 100.0
            line  = (f"ANOMALY: {label} ({conf:.1f}%)"
                    if (label in ANOMALY_SET or
                        (MAX_PERSONS is not None and label == "person" and counts.get("person",0) > MAX_PERSONS))
                    else f"Normal : {label} ({conf:.1f}%)")
            print(" " + line)
            logging.info(line)

        # If anomaly -> snapshot + CSV row
        snapshot_path = ""
        if is_anomaly:
            anomaly_count += 1
            last_reason = reason_txt
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(SNAPSHOT_DIR, f"anomaly_{ts_str}.jpg")
            try:
                saveImage(snapshot_path, img)
                logging.info("Saved snapshot: %s", snapshot_path)
            except Exception as e:
                logging.error("Failed to save snapshot: %s", e)
                snapshot_path = ""

            try:
                csv_writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    reason_txt,
                    ";".join(sorted(present)),
                    ";".join([f"{k}:{v}" for k,v in sorted(counts.items())]),
                    snapshot_path,
                ])
                csv_file.flush()
            except Exception as e:
                logging.error("CSV write failed: %s", e)

        # HUD/status
        status = "Anomaly!" if is_anomaly else "Normal"
        hud = f"{status} | anomalies={anomaly_count} | last={last_reason} | FPS={net.GetNetworkFPS():.0f}"
        display.Render(img)
        display.SetStatus(hud)

        if not display.IsStreaming():
            print("[EXIT] Output closed.")
            logging.info("Exit: output closed")
            break

except KeyboardInterrupt:
    print("\n[EXIT] Caught Ctrl+C, shutting down…")
    logging.info("Exit: Ctrl+C")
finally:
    try:
        csv_file.close()
    except Exception:
        pass
