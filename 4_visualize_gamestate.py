import json
import cv2
import os

# --- AYARLAR ---
# SNGS-021 sahnesinin yollarını belirliyoruz
BASE_DIR = "data/SoccerNet/gamestate-2025/valid/SNGS-021"
JSON_PATH = os.path.join(BASE_DIR, "Labels-GameState.json")
IMAGE_PATH = os.path.join(BASE_DIR, "img1", "000001.jpg") # İlk kare
OUTPUT_PATH = "test_gorsellestirme.jpg" # Çıktıyı kaydedeceğimiz dosya

def visualize_frame():
    # 1. Dosyaların varlığını kontrol et
    if not os.path.exists(JSON_PATH) or not os.path.exists(IMAGE_PATH):
        print("❌ JSON veya Resim dosyası bulunamadı! Yolları kontrol et.")
        return

    # 2. Resmi ve JSON'ı yükle
    print("Resim ve etiketler yükleniyor...")
    image = cv2.imread(IMAGE_PATH)
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    if not annotations:
        print("❌ JSON içinde 'annotations' bulunamadı.")
        return

    # 3. İlk resmin (000001.jpg) ID'sini bulalım
    # Genellikle ilk etiketin image_id'si, ilk resme aittir
    first_image_id = annotations[0]['image_id']
    print(f"Hedef Image ID: {first_image_id} - Çizim başlatılıyor...\n")

    # 4. Sadece bu resme ait olan etiketleri filtrele ve çiz
    draw_count = 0
    for ann in annotations:
        if ann['image_id'] == first_image_id:
            # Kutucuk (BBox) Koordinatları
            bbox = ann.get('bbox_image', {})
            x = int(bbox.get('x', 0))
            y = int(bbox.get('y', 0))
            w = int(bbox.get('w', 0))
            h = int(bbox.get('h', 0))

            # Nitelikler (Kimlik, Rol, Takım)
            attr = ann.get('attributes', {})
            role = attr.get('role', 'unknown')
            team = attr.get('team', 'unknown')
            jersey = attr.get('jersey', '')
            track_id = ann.get('track_id', '?')

            # Sadece oyuncuları çizelim (topu veya hakemi es geçebiliriz şimdilik)
            if role == "player":
                # Takıma göre renk belirle (BGR formatında)
                color = (0, 0, 255) if team == "left" else (255, 0, 0) # Sol takım Kırmızı, Sağ takım Mavi

                # 1. Kutucuğu Çiz
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                # 2. Bilgi Metnini Hazırla (ID ve Forma No)
                label_text = f"ID:{track_id} | No:{jersey}"
                
                # 3. Metni Çiz (Kutunun hemen üstüne)
                cv2.putText(image, label_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                draw_count += 1

    # 5. Sonucu Kaydet
    cv2.imwrite(OUTPUT_PATH, image)
    print(f"✅ İşlem tamam! Toplam {draw_count} oyuncu çizildi.")
    print(f"Görmek için proje ana dizinindeki '{OUTPUT_PATH}' dosyasını açabilirsin.")

if __name__ == "__main__":
    visualize_frame()