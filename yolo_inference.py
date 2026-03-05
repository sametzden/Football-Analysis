import torch
from ultralytics import YOLO


# Modeli indirelim (veya yüklü olanı açalım)
model = YOLO('models/best.pt') 

results = model.predict(source ='input_videos/C35bd9041_0 (22).mp4',save=True, show=True)

print(results[0])

for box in results[0].boxes:
    print(box)