"""
Kural Tabanlı Aksiyon Tespiti (Durum Değişimleri - State Transitions)
=====================================================================
Bu modül, hız/ivme hesaplamak yerine sadece topun kimde olduğu bilgisine
(sahip olma durumu) dayanır. "Top A oyuncusundan çıkıp B oyuncusuna gitti mi?"
mantığıyla çalışır.

Kurallar:
  1. PASS (Pas): Top Takım A oyuncusundan çıkıp yine Takım A oyuncusuna geçerse.
  2. INTERCEPTION (Pas Arası/Top Kaybı): Top Takım A'dan Takım B'ye geçerse.
  3. SHOT (Şut): Top Takım A oyuncusundan çıkıp karşı takımın ceza sahasına girerse
     veya karşı takım kalecisine giderse.
  4. DRIBBLE (Dribbling): Top uzun süre aynı oyuncuda kalır ve belli bir mesafe kat edilirse.
"""

import numpy as np
import cv2
import math
from utils.text_utils import put_text_tr

class ActionDetector:
    def __init__(self, fps=25, pitch_length=105, pitch_width=68):
        self.fps = fps
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # ── Parametreler ──
        self.contact_radius = 3.0       # metre — topun oyuncuya ait sayılması için max mesafe
        self.min_pass_frames = 5        # topun havalanıp/serbest kalma süresi (çok kısaysa pas sayılmaz, sekmeler filtrelenir)
        self.dribble_min_frames = 25    # 1 saniye aynı oyuncudaysa
        self.dribble_min_dist = 3.0     # en az 3 metre hareket etmeliki dribbling sayılsın
        
        # Ceza sahası x sınırları
        self.left_penalty_x = 16.5
        self.right_penalty_x = pitch_length - 16.5

        self.actions = []

    def analyze(self, tracks):
        self.actions = []
        num_frames = len(tracks["ball"])

        # Top pozisyonlarını çıkar
        ball_positions = []
        for f in range(num_frames):
            pos = tracks["ball"][f].get(1, {}).get("position_transformed")
            if pos is not None and len(pos) == 2:
                ball_positions.append(np.array(pos, dtype=float))
            else:
                ball_positions.append(None)

        last_holder = None
        free_ball_trajectory = []
        
        # Dribbling takibi için
        dribble_start_frame = -1
        dribble_start_pos = None

        for f in range(num_frames):
            ball_pos = ball_positions[f]
            if ball_pos is None:
                if last_holder is not None:
                    free_ball_trajectory.append((f, None))
                continue

            # Bu frame'de top kimde?
            current_holder = self._find_nearest_player(tracks, f, ball_pos)

            if current_holder is not None:
                c_id, c_pos, c_team, c_type = current_holder

                if last_holder is None:
                    # Top ilk defa birinde
                    last_holder = {
                        'id': c_id, 'team': c_team, 'type': c_type,
                        'frame_start': f, 'frame_last': f,
                        'pos_start': c_pos, 'pos_last': c_pos
                    }
                    dribble_start_frame = f
                    dribble_start_pos = c_pos
                    free_ball_trajectory = []
                    
                elif last_holder['id'] == c_id and last_holder['type'] == c_type:
                    # Aynı oyuncu topu sürmeye devam ediyor
                    last_holder['frame_last'] = f
                    last_holder['pos_last'] = c_pos
                    free_ball_trajectory = []
                    
                else:
                    # TOP SAHİBİ DEĞİŞTİ! (State Transition)
                    gap_frames = f - last_holder['frame_last']
                    
                    # Çok kısa süreli geçişler (örn: top kapışması) pas sayılmaz
                    if gap_frames >= 2:
                        action_type = None
                        
                        # Eğer topu kendi takım arkadaşı aldıysa, bu her zaman PAS'tır.
                        # (Ceza sahası içinde bile paslaşılabilir)
                        if c_team == last_holder['team']:
                            action_type = "PAS"
                        else:
                            # Topu rakip veya kaleci aldıysa
                            if c_type == "goalkeepers":
                                action_type = "ŞUT"
                            elif self._did_enter_penalty_box(free_ball_trajectory, last_holder['team']):
                                # Top rakip takıma geçti ama ceza sahası/kale bölgesinde olduysa
                                # Şut çekilmiş ve rakip defans bloklamış olabilir
                                action_type = "ŞUT"
                            else:
                                action_type = "PAS ARASI"
                        
                        self.actions.append({
                            "frame": last_holder['frame_last'],
                            "end_frame": f,
                            "type": action_type,
                            "player_team": last_holder['team'],
                            "receiver_team": c_team,
                            "ball_start": last_holder['pos_last'].tolist(),
                            "ball_end": c_pos.tolist(),
                            "display_until": f + 25
                        })
                    
                    # Dribbling kontrolü (önceki oyuncu için)
                    if last_holder['frame_last'] - dribble_start_frame >= self.dribble_min_frames:
                        dist = np.linalg.norm(last_holder['pos_last'] - dribble_start_pos)
                        if dist >= self.dribble_min_dist:
                            self.actions.append({
                                "frame": dribble_start_frame,
                                "type": "DRİBBLİNG",
                                "player_team": last_holder['team'],
                                "ball_start": dribble_start_pos.tolist(),
                                "ball_end": last_holder['pos_last'].tolist(),
                                "display_until": last_holder['frame_last'],
                                "distance": round(dist, 1)
                            })

                    # Yeni sahibe geç
                    last_holder = {
                        'id': c_id, 'team': c_team, 'type': c_type,
                        'frame_start': f, 'frame_last': f,
                        'pos_start': c_pos, 'pos_last': c_pos
                    }
                    dribble_start_frame = f
                    dribble_start_pos = c_pos
                    free_ball_trajectory = []
            else:
                # Top serbest (boşta)
                if last_holder is not None:
                    free_ball_trajectory.append((f, ball_pos))
                    
        # Video bittiğinde serbest kalan top şut olabilir mi?
        if last_holder is not None and len(free_ball_trajectory) > 5:
            if self._did_enter_penalty_box(free_ball_trajectory, last_holder['team']):
                self.actions.append({
                    "frame": last_holder['frame_last'],
                    "end_frame": free_ball_trajectory[-1][0],
                    "type": "ŞUT",
                    "player_team": last_holder['team'],
                    "receiver_team": "?",
                    "ball_start": last_holder['pos_last'].tolist(),
                    "ball_end": free_ball_trajectory[-1][1].tolist() if free_ball_trajectory[-1][1] is not None else last_holder['pos_last'].tolist(),
                    "display_until": num_frames - 1
                })

        # Kronolojik sırala
        self.actions.sort(key=lambda a: a["frame"])

        print(f"🎯 Aksiyon Tespiti (State Transition): {len(self.actions)} aksiyon bulundu")
        type_counts = {}
        for a in self.actions:
            type_counts[a["type"]] = type_counts.get(a["type"], 0) + 1
        for t, c in type_counts.items():
            print(f"   → {t}: {c}")

        return self.actions

    def _find_nearest_player(self, tracks, frame, ball_pos):
        """
        Verilen frame'de topa en yakın oyuncuyu/kaleciyi bulur.
        Returns: (track_id, position_array, team, type) or None
        """
        min_dist = float("inf")
        result = None

        for category in ["players", "goalkeepers"]:
            if frame >= len(tracks[category]):
                continue
            for track_id, data in tracks[category][frame].items():
                pos = data.get("position_transformed")
                if pos is None:
                    continue
                pos = np.array(pos, dtype=float)
                dist = np.linalg.norm(pos - ball_pos)
                if dist < min_dist and dist < self.contact_radius:
                    min_dist = dist
                    team = data.get("team")
                    result = (track_id, pos, team, category)

        return result

    def _did_enter_penalty_box(self, trajectory, attacking_team):
        """Topun serbest uçuş sırasında rakip ceza sahasına girip girmediğini kontrol eder."""
        if not trajectory:
            return False
            
        # Sahanın tamamını mı görüyoruz yoksa kırpılmış lokal bir alan mı?
        # (Örn. sabit perspektif modunda X max 23 metre falan oluyor)
        # Eğer bu trajectory'deki X değerleri çok darsa ve sahanın yarısından küçükse,
        # sabit küçük moddayız demektir, ceza sahası kuralı işlemez.
        valid_xs = [pos[0] for frame, pos in trajectory if pos is not None]
        if not valid_xs:
            return False
            
        # Eğer tüm video boyunca (veya bu pozisyonda) x hiçbir zaman 50'yi geçmiyorsa
        # (veya hep küçükse), bu tam saha değildir. Basit bir heuristik:
        if max(valid_xs) < 30 and min(valid_xs) >= -10:
            return False

        for x in valid_xs:
            if x < self.left_penalty_x or x > self.right_penalty_x:
                return True
        return False

    def get_summary(self):
        if not self.actions:
            return "Henüz aksiyon analizi yapılmadı."

        passes = [a for a in self.actions if a["type"] == "PAS"]
        shots = [a for a in self.actions if a["type"] == "ŞUT"]
        dribblings = [a for a in self.actions if a["type"] == "DRİBBLİNG"]
        interceptions = [a for a in self.actions if a["type"] == "PAS ARASI"]

        summary = (
            f"═══ AKSİYON ANALİZİ ÖZETİ (State Transitions) ═══\n"
            f"Toplam Pas: {len(passes)}\n"
            f"Toplam Pas Arası: {len(interceptions)}\n"
            f"Toplam Şut: {len(shots)}\n"
            f"Toplam Dribbling: {len(dribblings)}\n"
        )
        return summary

    def draw_actions(self, frames, actions=None):
        """
        Frame'ler üzerine aksiyon bilgisini çizer.
        """
        if actions is None:
            actions = self.actions

        frame_actions = {}
        for a in actions:
            start = a["frame"]
            end = a.get("display_until", start + 25)
            for f in range(start, min(end, len(frames))):
                if f not in frame_actions:
                    frame_actions[f] = []
                frame_actions[f].append(a)

        output_frames = []
        for frame_num, frame in enumerate(frames):
            if frame_num in frame_actions:
                # Aynı anda birden fazla aksiyon varsa sadece en güncelini çiz
                action = frame_actions[frame_num][-1]
                frame = self._draw_single_action(frame, action, frame_num)
            output_frames.append(frame)

        return output_frames

    def _draw_single_action(self, frame, action, current_frame):
        action_type = action["type"]
        team = action.get("player_team", "?")

        if action_type == "PAS":
            bg_color = (200, 120, 0)
            label = f"PAS (Takım {team})"
        elif action_type == "ŞUT":
            bg_color = (0, 0, 220)
            label = f"ŞUT! (Takım {team})"
        elif action_type == "DRİBBLİNG":
            bg_color = (0, 160, 0)
            label = f"Dribbling (Takım {team})"
        elif action_type == "PAS ARASI":
            bg_color = (0, 140, 255)
            label = f"PAS ARASI (Takım {action.get('receiver_team')})"
        else:
            return frame

        frame_h, frame_w = frame.shape[:2]
        banner_y = frame_h - 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, banner_y), (440, banner_y + 55), bg_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (10, banner_y), (440, banner_y + 55), (255, 255, 255), 1)

        frame = put_text_tr(frame, label,
                    (18, banner_y + 12), font_size=24,
                    color=(255, 255, 255))

        return frame
