"""
Futbol Analiz Pipeline — Tam Versiyon
======================================
SoccerNet GameState verileri + Football-Analysis modülleri ile
kapsamlı video analizi.

Pipeline:
  1. Video okuma
  2. YOLO tespiti + ByteTrack takibi
  3. Pozisyon ekleme
  4. Kamera hareketi tahmini ve telafisi
  5. Perspektif dönüşümü (piksel → metre)
  6. Top pozisyonu interpolasyonu
  7. Hız & mesafe hesabı
  8. Takım ataması (forma rengi ile)
  9. Top ataması (en yakın oyuncu)
  10. Top kaybı analizi (turnover)
  11. Kırık track birleştirme + interpolasyon
  12. Görselleştirme ve kaydetme
"""

import numpy as np
import os

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from ball_loss_analyzer import BallLossAnalyzer
from action_detector import ActionDetector
from utils.minimap_utils import MiniMap


# ── Ayarlar ──────────────────────────────────────────────────────────
MODEL_PATH = "best(2).pt"                    # 5 sınıf: player, ball, goalkeeper, referee, other
VIDEO_PATH = "SNGS-092_long_action.mp4"      # Analiz edilecek video
OUTPUT_PATH = "output_videos/analiz_sonuc_SNGS-092.avi"

# SoccerNet GameState (SADECE SoccerNet sahneleri için kullanılır. Özel video için None yapılmalı)
GAMESTATE_JSON = "data/SoccerNet/gamestate-2025/valid/SNGS-092/Labels-GameState.json"
START_FRAME_OFFSET = 0  # Baştan itibaren alıyoruz

# Stub dosyaları (cache — tekrar çalıştırırken hızlandırır)
USE_STUBS = False  # True yapılırsa önceki sonuçları diskten yükler
TRACK_STUB = "stubs/track_stubs_092.pkl"
CAMERA_STUB = "stubs/camera_movement_stubs_092.pkl"


def main():
    print("=" * 60)
    print("⚽ FUTBOL ANALİZ PİPELINE — BAŞLATILIYOR")
    print("=" * 60)

    # ── 1. Video Okuma ───────────────────────────────────────────
    print("\n📹 [1/12] Video okunuyor...")
    video_frames = read_video(VIDEO_PATH)
    print(f"   → {len(video_frames)} frame yüklendi "
          f"({video_frames[0].shape[1]}x{video_frames[0].shape[0]})")

    # ── 2. Tespit + Takip ────────────────────────────────────────
    print("\n🔍 [2/12] YOLO tespiti + ByteTrack takibi başlatılıyor...")
    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=USE_STUBS,
        stub_path=TRACK_STUB
    )
    print(f"   → Players: {sum(len(t) for t in tracks['players'])} detection")
    print(f"   → Goalkeepers: {sum(len(t) for t in tracks['goalkeepers'])} detection")
    print(f"   → Referees: {sum(len(t) for t in tracks['referees'])} detection")
    print(f"   → Ball: {sum(len(t) for t in tracks['ball'])} detection")

    # ── 3. Pozisyon Ekleme ───────────────────────────────────────
    print("\n📍 [3/12] Pozisyon bilgisi ekleniyor...")
    tracker.add_position_to_tracks(tracks)

    # ── 4. Kamera Hareketi ───────────────────────────────────────
    print("\n📷 [4/12] Kamera hareketi hesaplanıyor...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=USE_STUBS,
        stub_path=CAMERA_STUB
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    print("   → Kamera hareketi telafisi tamamlandı")

    # ── 5. Perspektif Dönüşümü ───────────────────────────────────
    print("\n📐 [5/12] Saha perspektif dönüşümü...")
    if GAMESTATE_JSON and os.path.exists(GAMESTATE_JSON):
        view_transformer = ViewTransformer(gamestate_json_path=GAMESTATE_JSON, frame_offset=START_FRAME_OFFSET)
    else:
        view_transformer = ViewTransformer()  # Varsayılan sabit noktalar
    view_transformer.add_transformed_position_to_tracks(tracks)

    # ── 6. Top İnterpolasyonu ────────────────────────────────────
    print("\n🏐 [6/12] Top pozisyonu interpolasyonu...")
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])
    print("   → Eksik top pozisyonları dolduruldu")

    # İnterpolasyon topun position/adjusted/transformed bilgisini sildiği için
    # yeniden hesaplıyoruz (sadece top için)
    tracker.add_position_to_tracks({"ball": tracks["ball"]})
    camera_movement_estimator.add_adjust_positions_to_tracks(
        {"ball": tracks["ball"]}, camera_movement_per_frame)
    view_transformer.add_transformed_position_to_tracks({"ball": tracks["ball"]})

    # ── 7. Hız & Mesafe ──────────────────────────────────────────
    print("\n🏃 [7/12] Hız ve mesafe hesaplanıyor...")
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # ── 8. Takım Ataması ─────────────────────────────────────────
    print("\n👕 [8/12] Takım ataması yapılıyor (forma rengi)...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    frame_width = video_frames[0].shape[1]

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = \
                team_assigner.team_colors[team]

    # Kaleci ataması (konum bazlı)
    for frame_num, goalkeeper_track in enumerate(tracks["goalkeepers"]):
        for goalkeeper_id, track in goalkeeper_track.items():
            team = team_assigner.get_goalkeeper_team(
                video_frames[frame_num], track["bbox"], goalkeeper_id, frame_width)
            tracks["goalkeepers"][frame_num][goalkeeper_id]["team"] = team
            tracks["goalkeepers"][frame_num][goalkeeper_id]["team_color"] = \
                team_assigner.team_colors[team]

    print(f"   → Takım 1 rengi: {team_assigner.team_colors[1]}")
    print(f"   → Takım 2 rengi: {team_assigner.team_colors[2]}")

    # ── 9. Top Ataması ───────────────────────────────────────────
    print("\n⚽ [9/12] Top kontrolü atanıyor...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox", [])
        if not ball_bbox:
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else 1)
            continue

        assigned_player = player_assigner.assign_ball_to_players(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else 1)

    team_ball_control = np.array(team_ball_control)

    team1_pct = (team_ball_control == 1).sum() / len(team_ball_control) * 100
    team2_pct = (team_ball_control == 2).sum() / len(team_ball_control) * 100
    print(f"   → Takım 1 Top Kontrolü: %{team1_pct:.1f}")
    print(f"   → Takım 2 Top Kontrolü: %{team2_pct:.1f}")

    # ── 10. Aksiyon Tespiti (Pas / Şut / Dribbling) ──────────────
    print("\n🎯 [10/13] Aksiyon tespiti (Pas / Şut / Dribbling)...")
    action_detector = ActionDetector(fps=25, pitch_length=105, pitch_width=68)
    detected_actions = action_detector.analyze(tracks)
    print(action_detector.get_summary())

    # ── 11. Top Kaybı Analizi ────────────────────────────────────
    print("\n🔄 [11/13] Top kaybı analizi...")
    ball_loss_analyzer = BallLossAnalyzer(min_possession_frames=5)
    turnovers = ball_loss_analyzer.analyze(tracks, team_ball_control)
    print(ball_loss_analyzer.get_summary())

    # ── 12. Kırık Track Birleştirme + İnterpolasyon ──────────────
    print("\n🔗 [12/13] Kırık track'ler birleştiriliyor ve eksik frame'ler dolduruluyor...")
    tracker.merge_fragmented_tracks(tracks)
    tracker.interpolate_missing_frames(tracks)
    print("   → Track birleştirme ve interpolasyon tamamlandı")

    # ── 13. Görselleştirme ve Kaydetme ───────────────────────────
    print("\n🎨 [13/13] Çıktı videosu oluşturuluyor...")

    # Ana anotasyonlar (elips, üçgen, takım rengi, top kontrol)
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control)

    # Kamera hareketi bilgisi
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame)

    # 2D Minimap Çizimi (+ aksiyon okları)
    minimap = MiniMap()
    for frame_num, frame in enumerate(output_video_frames):
        output_video_frames[frame_num] = minimap.draw(
            frame, tracks, frame_num, team_assigner,
            actions=detected_actions
        )

    # Aksiyon banner'ları (PAS / ŞUT / DRİBBLİNG)
    output_video_frames = action_detector.draw_actions(output_video_frames)

    # Top kaybı işaretleri
    #output_video_frames = ball_loss_analyzer.draw_turnover_markers(
    #    output_video_frames, turnovers)

    # Videoyu kaydet
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    save_video(output_video_frames, OUTPUT_PATH)

    print(f"\n{'=' * 60}")
    print(f"✅ ANALİZ TAMAMLANDI!")
    print(f"   Çıktı: {OUTPUT_PATH}")
    print(f"   Frame sayısı: {len(output_video_frames)}")
    print(f"   Toplam aksiyon: {len(detected_actions)}")
    print(f"   Toplam turnover: {len(turnovers)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()