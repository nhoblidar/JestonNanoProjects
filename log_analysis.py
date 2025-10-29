import csv, os, collections, datetime

BASE = "/jetson-inference/data/nandini"
CSV_PATH = os.path.join(BASE, "anomaly_log.csv")
IMG_DIR  = os.path.join(BASE, "anomaly_images")

total_rows = 0
reasons = collections.Counter()
class_counts = collections.Counter()
by_minute = collections.Counter()

with open(CSV_PATH, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        total_rows += 1
        reasons[row["reason"]] += 1

        # parse “counts” like "laptop:1;person:2"
        for kv in row["counts"].split(";"):
            if not kv: continue
            k, v = kv.split(":")
            class_counts[k] += int(v)

        # minute bucket
        t = datetime.datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        by_minute[t.strftime("%Y-%m-%d %H:%M")] += 1

snapshots = len([p for p in os.listdir(IMG_DIR) if p.lower().endswith(".jpg")])

print("=== Anomaly Report ===")
print(f"CSV anomalies:        {total_rows}")
print(f"Snapshot files:       {snapshots}")
print("\nTop reasons:")
for reason, n in reasons.most_common(10):
    print(f"  {n:4d}  {reason}")

print("\nClass totals in anomaly frames:")
for cls, n in sorted(class_counts.items()):
    print(f"  {cls:12s} {n}")

print("\nAnomalies by minute (top 10):")
for minute, n in sorted(by_minute.items(), key=lambda x: (-x[1], x[0]))[:10]:
    print(f"  {minute}  {n}")
