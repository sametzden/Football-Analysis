from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import pickle
import os
import cv2
import math
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from utils.text_utils import put_text_tr


class Tracker:
    def __init__(self, model_path):
        """YOLO modelini başlatır. Takip işlemi model.track ile BoTSORT üzerinden yapılacaktır."""
        self.model = YOLO(model_path)

    # ── Pozisyon Ekleme ──────────────────────────────────────────────
    def add_position_to_tracks(self, tracks):
        """Her nesneye pozisyon bilgisi ekler (top: merkez, diğer: ayak)."""
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_name == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_name][frame_num][track_id]['position'] = position

    # ── Top Pozisyonu İnterpolasyonu ─────────────────────────────────
    def interpolate_ball_position(self, ball_positions):
        """Eksik top konumlarını lineer interpolasyonla doldurur."""
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Eksik değerleri interpole et
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    # ── Frame Tespiti ────────────────────────────────────────────────
    def detect_frames(self, frames):
        """Frame'leri batch'ler halinde YOLO modeline gönderir ve BoTSORT ile takip eder."""
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # BoTSORT kullanarak ID takibi yapıyoruz (persist=True ile batch'ler arası hafıza korunur)
            detections_batch = self.model.track(frames[i:i + batch_size], tracker="botsort.yaml", persist=True, conf=0.25, imgsz=640)
            detections += detections_batch
        return detections

    # ── Nesne Takibi ─────────────────────────────────────────────────
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Tüm frame'leri işleyip her nesne türü için track dictionary döndürür.
        Yapı: {players: [{track_id: {bbox: ...}}, ...], goalkeepers: [...], ...}
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "goalkeepers": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            tracks["players"].append({})
            tracks["goalkeepers"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            if detection.boxes is not None:
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

                for i, box in enumerate(detection.boxes):
                    bbox = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    
                    if cls_id == cls_names_inv.get("ball"):
                        # Top için ID takibi genelde başarısız olur, her karede en iyi tespit yeterlidir
                        tracks["ball"][frame_num][1] = {"bbox": bbox}
                    else:
                        # Eğer nesneye henüz bir track_id atanamadıysa yoksay
                        track_id = int(box.id[0].item()) if box.id is not None else None
                        if track_id is None:
                            continue
                            
                        if cls_id == cls_names_inv.get("player"):
                            tracks["players"][frame_num][track_id] = {"bbox": bbox}
                        elif cls_id == cls_names_inv.get("goalkeeper"):
                            tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}
                        elif cls_id == cls_names_inv.get("referee"):
                            tracks["referees"][frame_num][track_id] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    # ── Kırık Track'leri Birleştirme ─────────────────────────────────
    def merge_fragmented_tracks(self, tracks, max_frame_gap=60, max_distance=150):
        """
        Oklüzyon sonrası kırılan tracklet'leri birleştirir.
        Temporal yakınlık, uzaysal mesafe ve takım tutarlılığını kontrol eder.
        """
        for category in ["players", "goalkeepers", "referees"]:
            track_history = {}

            # Adım 1: Track geçmişini topla
            for frame_num, frame_tracks in enumerate(tracks[category]):
                for track_id, track_data in frame_tracks.items():
                    pos = track_data.get("position", [0, 0])
                    team = track_data.get("team")

                    if track_id not in track_history:
                        track_history[track_id] = {
                            "start_frame": frame_num,
                            "end_frame": frame_num,
                            "start_pos": pos,
                            "end_pos": pos,
                            "team": team
                        }
                    else:
                        track_history[track_id]["end_frame"] = frame_num
                        track_history[track_id]["end_pos"] = pos
                        if track_history[track_id]["team"] is None and team is not None:
                            track_history[track_id]["team"] = team

            # Adım 2: Greedy birleştirme
            sorted_tracks = sorted(track_history.keys(),
                                   key=lambda t: track_history[t]["start_frame"])
            id_mapping = {}

            for i, current_id in enumerate(sorted_tracks):
                current_data = track_history[current_id]
                best_match_id = None
                best_match_distance = float('inf')

                for j in range(i - 1, -1, -1):
                    prev_id = sorted_tracks[j]

                    # Mapping'i root'a kadar izle
                    root_prev_id = prev_id
                    while root_prev_id in id_mapping:
                        root_prev_id = id_mapping[root_prev_id]

                    prev_data = track_history[root_prev_id]
                    frame_gap = current_data["start_frame"] - prev_data["end_frame"]

                    if 0 < frame_gap <= max_frame_gap:
                        dist = math.hypot(
                            current_data["start_pos"][0] - prev_data["end_pos"][0],
                            current_data["start_pos"][1] - prev_data["end_pos"][1]
                        )
                        if dist <= max_distance and dist < best_match_distance:
                            # Takım eşleşmesi kontrolü
                            if current_data["team"] is not None and prev_data["team"] is not None:
                                if current_data["team"] == prev_data["team"]:
                                    best_match_id = root_prev_id
                                    best_match_distance = dist
                            else:
                                best_match_id = root_prev_id
                                best_match_distance = dist

                if best_match_id is not None:
                    id_mapping[current_id] = best_match_id
                    track_history[best_match_id]["end_frame"] = current_data["end_frame"]
                    track_history[best_match_id]["end_pos"] = current_data["end_pos"]

            # Adım 3: ID mapping'i uygula
            for frame_num, frame_tracks in enumerate(tracks[category]):
                new_frame_tracks = {}
                for track_id, track_data in frame_tracks.items():
                    root_id = track_id
                    while root_id in id_mapping:
                        root_id = id_mapping[root_id]
                    new_frame_tracks[root_id] = track_data
                tracks[category][frame_num] = new_frame_tracks

    # ── Eksik Frame'leri İnterpolasyon ───────────────────────────────
    def interpolate_missing_frames(self, tracks):
        """
        Track birleştirmesi sonrası boş kalan frame'leri lineer
        interpolasyonla doldurur.
        """
        for category in ["players", "goalkeepers", "referees"]:
            track_dict = {}
            for frame_num, frame_tracks in enumerate(tracks[category]):
                for track_id, track_data in frame_tracks.items():
                    if track_id not in track_dict:
                        track_dict[track_id] = {}
                    track_dict[track_id][frame_num] = track_data

            for track_id, frames_data in track_dict.items():
                sorted_frames = sorted(frames_data.keys())
                if not sorted_frames:
                    continue

                start_frame = sorted_frames[0]
                end_frame = sorted_frames[-1]

                for f in range(start_frame + 1, end_frame):
                    if f not in frames_data:
                        prev_f = max(k for k in sorted_frames if k < f)
                        next_f = min(k for k in sorted_frames if k > f)

                        prev_data = frames_data[prev_f]
                        next_data = frames_data[next_f]
                        ratio = (f - prev_f) / (next_f - prev_f)

                        interpolated_data = {}

                        def interpolate_array(arr1, arr2):
                            if arr1 is None or arr2 is None:
                                return arr1
                            return [a + (b - a) * ratio for a, b in zip(arr1, arr2)]

                        # Standart alanlar
                        interpolated_data["bbox"] = interpolate_array(
                            prev_data.get("bbox"), next_data.get("bbox"))
                        interpolated_data["position"] = interpolate_array(
                            prev_data.get("position"), next_data.get("position"))

                        # Dönüştürülmüş pozisyonlar
                        if "position_transformed" in prev_data and "position_transformed" in next_data:
                            interpolated_data["position_transformed"] = interpolate_array(
                                prev_data["position_transformed"],
                                next_data["position_transformed"])

                        # Kamera-düzeltilmiş pozisyonlar
                        if "position_adjusted" in prev_data and "position_adjusted" in next_data:
                            interpolated_data["position_adjusted"] = interpolate_array(
                                prev_data["position_adjusted"],
                                next_data["position_adjusted"])

                        # Statik metadata'yı kopyala
                        for key in ["team", "team_color", "has_ball"]:
                            if key in prev_data:
                                interpolated_data[key] = prev_data[key]

                        tracks[category][f][track_id] = interpolated_data

    # ── Çizim: Elips ─────────────────────────────────────────────────
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """Bounding box altına elips ve ID etiketi çizer."""
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(int(x_center), int(y2)),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    # ── Çizim: Üçgen (Top İşareti) ──────────────────────────────────
    def draw_triangle(self, frame, bbox, color):
        """Bounding box üstüne üçgen çizer (top kontrolü göstergesi)."""
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    # ── Çizim: Takım Top Kontrolü ────────────────────────────────────
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """Frame üzerine takım top kontrol yüzdesini çizer."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 840), (1910, 975), (20, 20, 20), -1)
        alpha = 0.55
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        total = team1_num_frames + team2_num_frames
        if total > 0:
            team1 = team1_num_frames / total
            team2 = team2_num_frames / total
        else:
            team1 = team2 = 0.5

        frame = put_text_tr(frame, f"Takim 1 Top Kontrolu: {team1:.1%}",
                    (1360, 858), font_size=20, color=(255, 255, 255))
        frame = put_text_tr(frame, f"Takim 2 Top Kontrolu: {team2:.1%}",
                    (1360, 900), font_size=20, color=(255, 255, 255))

        return frame

    # ── Çizim: Ana Anotasyon ─────────────────────────────────────────
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """Tüm frame'lere elips, üçgen, takım rengi ve top kontrol bilgisi çizer."""
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            goalkeeper_dict = tracks["goalkeepers"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Oyuncuları çiz
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (255, 0, 0))

            # Kalecileri çiz (beyaz halka ile ayırt)
            for track_id, goalkeeper in goalkeeper_dict.items():
                color = goalkeeper.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, goalkeeper["bbox"], color)
                frame = self.draw_ellipse(frame, goalkeeper["bbox"], (255, 255, 255))

                if goalkeeper.get("has_ball", False):
                    frame = self.draw_triangle(frame, goalkeeper["bbox"], (255, 0, 0))

            # Hakemleri çiz
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Topu çiz
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Takım top kontrol istatistiği
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_video_frames.append(frame)

        return output_video_frames
