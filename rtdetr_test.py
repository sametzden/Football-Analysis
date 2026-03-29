"""
rtdetr_test.py
--------------
RT-DETR modelini bir video üzerinde test eder ve YOLO ile yan yana karşılaştırır.

Kullanım:
    cd /home/samet/Projects/football_analysis
    source cv_env/bin/activate
    python rtdetr_test.py
"""

import cv2
import numpy as np
from ultralytics import YOLO, RTDETR
import time
import os

# ─── Ayarlar ──────────────────────────────────────────────────────────
RTDETR_MODEL = "models/rtdetr_best.pt"
YOLO_MODEL   = "models/best.pt"
VIDEO_PATH   = "input_videos/test (20).mp4"
OUTPUT_DIR   = "output_videos"
CONF         = 0.3
MAX_FRAMES   = 200   # Kaç frame işlensin (None = tamamı)
CLASS_COLORS = {
    "ball":       (0, 255, 0),    # Yeşil
    "goalkeeper": (255, 165, 0),  # Turuncu
    "player":     (0, 120, 255),  # Mavi
    "referee":    (0, 255, 255),  # Sarı
}
# ──────────────────────────────────────────────────────────────────────


def draw_detections(frame, results, model_name, fps):
    """Detections + model adı + FPS bilgisini frame'e yazar."""
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        cls_name = results[0].names[cls_id]
        color    = CLASS_COLORS.get(cls_name, (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Model adı + FPS overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 50), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"{model_name}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}  conf>={CONF}", (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    return frame


def run_test():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== RT-DETR vs YOLO Video Test ===\n")

    print("Modeller yükleniyor...")
    rtdetr = RTDETR(RTDETR_MODEL)
    yolo   = YOLO(YOLO_MODEL)
    print("  RT-DETR → OK")
    print("  YOLO    → OK\n")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {VIDEO_PATH}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 24

    # 2 model yan yana → genişlik 2x
    out_path = os.path.join(OUTPUT_DIR, "rtdetr_vs_yolo.avi")
    writer   = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        FPS,
        (W * 2, H)
    )

    frame_num = 0
    rtdetr_times, yolo_times = [], []

    print(f"Video işleniyor: {VIDEO_PATH}")
    print(f"Çözünürlük: {W}x{H}  FPS: {FPS}  Max frames: {MAX_FRAMES or 'tümü'}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if MAX_FRAMES and frame_num >= MAX_FRAMES:
            break

        # ─── RT-DETR inference ───
        t0 = time.perf_counter()
        r_rtdetr = rtdetr.predict(frame, conf=CONF, verbose=False, imgsz=1280)
        rtdetr_ms = (time.perf_counter() - t0) * 1000
        rtdetr_times.append(rtdetr_ms)

        # ─── YOLO inference ───
        t0 = time.perf_counter()
        r_yolo = yolo.predict(frame, conf=CONF, verbose=False, imgsz=1280)
        yolo_ms = (time.perf_counter() - t0) * 1000
        yolo_times.append(yolo_ms)

        # ─── Annotate ───
        left  = draw_detections(frame.copy(), r_rtdetr,
                                "RT-DETR-L (fine-tuned)", 1000 / rtdetr_ms)
        right = draw_detections(frame.copy(), r_yolo,
                                "YOLOv8 (fine-tuned)", 1000 / yolo_ms)

        combined = np.hstack([left, right])
        writer.write(combined)

        if frame_num % 30 == 0:
            n_rtdetr = len(r_rtdetr[0].boxes)
            n_yolo   = len(r_yolo[0].boxes)
            print(f"  Frame {frame_num:4d} | RT-DETR: {n_rtdetr:2d} det, {rtdetr_ms:5.1f}ms"
                  f"  |  YOLO: {n_yolo:2d} det, {yolo_ms:5.1f}ms")

        frame_num += 1

    cap.release()
    writer.release()

    # ─── Özet ───
    def avg(lst): return sum(lst) / len(lst) if lst else 0

    print(f"\n{'═'*55}")
    print(f"{'Metrik':<25} {'RT-DETR':>12}  {'YOLO':>12}")
    print(f"{'─'*55}")
    print(f"{'Toplam frame':<25} {frame_num:>12}  {frame_num:>12}")
    print(f"{'Ort. inference (ms)':<25} {avg(rtdetr_times):>12.1f}  {avg(yolo_times):>12.1f}")
    print(f"{'Ort. FPS':<25} {1000/avg(rtdetr_times):>12.1f}  {1000/avg(yolo_times):>12.1f}")
    print(f"{'═'*55}")
    print(f"\nKaydedildi → {out_path}")


if __name__ == "__main__":
    run_test()
