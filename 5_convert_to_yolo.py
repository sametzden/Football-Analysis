import json
import os
import shutil
import glob

# --- AYARLAR ---
VALID_DIR = "data/SoccerNet/gamestate-2025/valid"

# YOLO için Klasör Hiyerarşisi
OUTPUT_BASE = "yolo_dataset"
OUT_IMAGES_VAL = os.path.join(OUTPUT_BASE, "images", "val")
OUT_LABELS_VAL = os.path.join(OUTPUT_BASE, "labels", "val")

# Hedef klasörleri sıfırdan temizleyip oluşturmak istersen (isteğe bağlı)
os.makedirs(OUT_IMAGES_VAL, exist_ok=True)
os.makedirs(OUT_LABELS_VAL, exist_ok=True)

def convert_all_to_yolo():
    print("🚀 Tüm Sahneler İçin Dönüşüm Başlıyor...\n")

    # VALID klasörü içindeki tüm SNGS- ile başlayan klasörleri bul
    scene_folders = sorted([f for f in os.listdir(VALID_DIR) if f.startswith("SNGS-")])
    print(f"Toplam {len(scene_folders)} adet sahne (sequence) bulundu.")

    total_images_processed = 0

    # Her bir sahne klasörünün içine tek tek giriyoruz
    for scene in scene_folders:
        scene_dir = os.path.join(VALID_DIR, scene)
        json_path = os.path.join(scene_dir, "Labels-GameState.json")
        img_dir = os.path.join(scene_dir, "img1")

        if not os.path.exists(json_path):
            continue

        print(f"İşleniyor: {scene}...")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Resim Bilgileri
        image_dict = {}
        for img in data.get('images', []):
            image_dict[img['image_id']] = {
                'file_name': img.get('file_name', f"{str(img['image_id'])[-6:]}.jpg"),
                'width': img.get('width', 1920),
                'height': img.get('height', 1080)
            }

        annotations = data.get('annotations', [])
        labels_by_image = {}

        # Etiketleri YOLO Formatına Çevir
        for ann in annotations:
            if ann.get('attributes', {}).get('role') == 'player':
                img_id = ann['image_id']
                if img_id not in labels_by_image:
                    labels_by_image[img_id] = []
                
                bbox = ann['bbox_image']
                img_info = image_dict.get(img_id, {'width': 1920, 'height': 1080})
                
                w_img = img_info['width']
                h_img = img_info['height']

                # Normalizasyon
                x_center_norm = bbox['x_center'] / w_img
                y_center_norm = bbox['y_center'] / h_img
                w_norm = bbox['w'] / w_img
                h_norm = bbox['h'] / h_img

                yolo_line = f"0 {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                labels_by_image[img_id].append(yolo_line)

        # Dosyaları Kopyala ve .txt Oluştur
        for img_id, yolo_lines in labels_by_image.items():
            img_info = image_dict.get(img_id)
            if not img_info: continue
            
            # Tüm klasörlerdeki resim isimleri 000001.jpg olduğu için çakışma olmasın diye
            # Yeni isim: SNGS-021_000001.jpg yapıyoruz
            original_file_name = img_info['file_name']
            new_file_name = f"{scene}_{original_file_name}"
            txt_file_name = new_file_name.replace(".jpg", ".txt")
            
            src_img_path = os.path.join(img_dir, original_file_name)
            dst_img_path = os.path.join(OUT_IMAGES_VAL, new_file_name)
            dst_txt_path = os.path.join(OUT_LABELS_VAL, txt_file_name)
            
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
            
            with open(dst_txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))
                
            total_images_processed += 1

    print(f"\n✅ MUHTEŞEM! Toplam {total_images_processed} resim ve etiket YOLO formatına çevrildi.")

if __name__ == "__main__":
    convert_all_to_yolo()