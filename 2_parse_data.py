import json
import os

# Çıkardığın JSON dosyasının tam yolunu buraya yaz
# ÖRNEK: "data/SoccerNet/spotting-ball-2025/valid/england_epl/2016-2017/mac_adi/Labels-v2.json"
JSON_PATH = "data/SoccerNet/spotting-ball-2025/valid/england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End/Labels-ball.json" 

def explore_json(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Dosya bulunamadı: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("✅ JSON başarıyla yüklendi!\n")
    print("-" * 40)
    print(f"Oynanan Maç: {data.get('UrlLocal', 'Bilinmiyor')}")
    print("-" * 40)
    
    # Sadece ilk 3 aksiyonu (etiketi) inceleyelim ki ekran dolup taşmasın
    annotations = data.get('annotations', [])
    print(f"Toplam Aksiyon Sayısı: {len(annotations)}\n")
    
    for i, action in enumerate(annotations[:3]):
        print(f"--- Aksiyon {i+1} ---")
        # JSON'ın içindeki anahtarları (keys) doğrudan yazdırıyoruz
        for key, value in action.items():
            print(f"  {key}: {value}")
        print("\n")

if __name__ == "__main__":
    explore_json(JSON_PATH)