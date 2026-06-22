import os
from SoccerNet.Downloader import SoccerNetDownloader

DATA_DIR = "data/SoccerNet"
os.makedirs(DATA_DIR, exist_ok=True)

# İndirici nesnesi başlatılıyor
downloader = SoccerNetDownloader(LocalDirectory=DATA_DIR)
downloader.password = "s0cc3rn3t"

try:
    print("🚀 SN-BAS-2025 (Ball Action Spotting) HuggingFace'ten çekiliyor...")
    
    # Senin eski ve çalışan taktiğindeki o sihirli parametreyi (source="HuggingFace") kullanıyoruz
    downloader.downloadDataTask(
        task="spotting-ball-2025", 
        split=["train", "valid", "test"], 
        source="HuggingFace"
    )
    
    print("\n✅ İndirme tamamlandı! Eski sunucu çökmelerinden tamamen kurtulduk.")
except Exception as e:
    print(f"❌ Bir hata oluştu: {e}")