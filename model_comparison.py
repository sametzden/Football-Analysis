"""
model_comparison.py
-------------------
YOLO vs RT-DETR karşılaştırma scripti.
Validation seti üzerinde mAP, FPS ve sınıf bazlı AP metriklerini hesaplar.

Kullanım:
    python model_comparison.py \
        --yolo   models/best.pt \
        --rtdetr models/rtdetr_best.pt \
        --data   training/football-players-detection-1/data.yaml \
        --imgsz  1280

Eğer RT-DETR modeli henüz eğitilmediyse sadece YOLO çalıştırmak için:
    python model_comparison.py --yolo models/best.pt --data training/.../data.yaml
"""

import argparse
import time
import csv
import os
import cv2
import numpy as np

from ultralytics import YOLO, RTDETR

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CLASS_NAMES = ["ball", "goalkeeper", "player", "referee"]
OUTPUT_CSV  = "output_videos/comparison_results.csv"


def load_model(path: str):
    """Loads YOLO or RT-DETR based on filename."""
    if "rtdetr" in path.lower():
        print(f"  Loading RT-DETR: {path}")
        return RTDETR(path)
    else:
        print(f"  Loading YOLO:    {path}")
        return YOLO(path)


def evaluate_model(model, data_yaml: str, imgsz: int) -> dict:
    """Runs validation and returns metrics dict."""
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        verbose=False,
        save=False
    )
    metrics = {
        "mAP50":    round(float(results.box.map50),  4),
        "mAP50-95": round(float(results.box.map),    4),
    }
    # Per-class AP50
    if hasattr(results.box, "ap_class_index") and results.box.ap_class_index is not None:
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
            metrics[f"AP50_{cls_name}"] = round(float(results.box.ap50[i]), 4)
    return metrics


def benchmark_fps(model, imgsz: int, n_frames: int = 100) -> float:
    """Measures inference FPS using a blank frame."""
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    # Warmup
    for _ in range(5):
        model.predict(dummy, verbose=False)
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_frames):
        model.predict(dummy, verbose=False)
    elapsed = time.perf_counter() - start
    return round(n_frames / elapsed, 2)


def print_table(rows: list[dict], models: list[str]):
    """Prints a formatted comparison table to terminal."""
    all_keys = list(rows[0].keys())
    col_w = max(len(k) for k in all_keys) + 2

    header = f"{'Metric':<{col_w}}" + "".join(f"{m:<18}" for m in models)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for key in all_keys:
        row = f"{key:<{col_w}}"
        for r in rows:
            val = str(r.get(key, "N/A"))
            row += f"{val:<18}"
        print(row)
    print("=" * len(header) + "\n")


def save_csv(rows: list[dict], models: list[str], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    all_keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + models)
        for key in all_keys:
            writer.writerow([key] + [r.get(key, "N/A") for r in rows])
    print(f"Results saved → {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLO vs RT-DETR Model Comparison")
    parser.add_argument("--yolo",   required=True,            help="Path to YOLO model (.pt)")
    parser.add_argument("--rtdetr", default=None,             help="Path to RT-DETR model (.pt) — optional")
    parser.add_argument("--data",   required=True,            help="Path to data.yaml")
    parser.add_argument("--imgsz",  type=int, default=1280,   help="Inference image size (default: 1280)")
    parser.add_argument("--fps-frames", type=int, default=50, help="Number of frames for FPS benchmark")
    args = parser.parse_args()

    model_paths = [args.yolo]
    if args.rtdetr:
        model_paths.append(args.rtdetr)
    model_names = [os.path.basename(p) for p in model_paths]

    all_metrics = []
    for path in model_paths:
        print(f"\n{'─'*50}")
        model = load_model(path)

        print("  → Running validation...")
        metrics = evaluate_model(model, args.data, args.imgsz)

        print("  → Benchmarking FPS...")
        fps = benchmark_fps(model, args.imgsz, args.fps_frames)
        metrics["FPS"] = fps

        all_metrics.append(metrics)
        del model  # free memory before loading next model

    # Merge all keys
    all_keys = []
    for m in all_metrics:
        for k in m:
            if k not in all_keys:
                all_keys.append(k)
    # Normalize missing keys
    for m in all_metrics:
        for k in all_keys:
            m.setdefault(k, "N/A")

    print_table(all_metrics, model_names)
    save_csv(all_metrics, model_names, OUTPUT_CSV)


if __name__ == "__main__":
    main()
