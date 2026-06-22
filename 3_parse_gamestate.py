import json
import os

# Sadece ilk sahne olan SNGS-021'in etiket dosyasına odaklanıyoruz
JSON_PATH = "data/SoccerNet/gamestate-2025/valid/SNGS-021/Labels-GameState.json" 

def explore_gamestate(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Dosya bulunamadı. Yolun doğruluğunu kontrol et: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("✅ GameState JSON başarıyla yüklendi!\n")
    print("-" * 50)
    print("Dosya İçindeki Ana Bölümler (Keys):", list(data.keys()))
    print("-" * 50)
    
    # Genellikle bu tarz veriler 'annotations' (etiketler) anahtarı altında tutulur
    if 'annotations' in data:
        print(f"Toplam Etiket Sayısı: {len(data['annotations'])}\n")
        print("--- Örnek Veri (İlk Eleman) ---")
        # İlk etiket objesini güzel bir formatta yazdırıyoruz
        print(json.dumps(data['annotations'][0], indent=4, ensure_ascii=False))
    else:
        # Eğer yapı farklıysa, ilk anahtarın içeriğine bakalım
        first_key = list(data.keys())[0]
        print(f"'{first_key}' anahtarı altındaki ilk veri:")
        print(json.dumps(data[first_key][0] if isinstance(data[first_key], list) else data[first_key], indent=4, ensure_ascii=False))

if __name__ == "__main__":
    explore_gamestate(JSON_PATH)