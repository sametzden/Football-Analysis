import numpy as np


class BallLossAnalyzer:
    """
    Top kaybı (turnover) anlarını tespit ve analiz eder.
    
    Top kontrolü bir takımdan diğerine geçtiğinde:
    - Kaybı yapan frame, takım ve oyuncu kaydedilir
    - Top kaybı sonrası karşı takımın tepki süresi analiz edilir
    """

    def __init__(self, min_possession_frames=5):
        """
        Args:
            min_possession_frames: Bir top geçişinin "gerçek" sayılması için
                                   minimum top tutma süresi (frame cinsinden).
                                   Kısa sıçramaları filtreler.
        """
        self.min_possession_frames = min_possession_frames
        self.turnovers = []

    def analyze(self, tracks, team_ball_control):
        """
        Top kaybı anlarını tespit eder.
        
        Args:
            tracks: Tracker'dan gelen track dictionary
            team_ball_control: np.array — her frame için topun hangi takımda olduğu (1 veya 2)
            
        Returns:
            list of dict: Her turnover için {frame, losing_team, gaining_team, 
                          losing_player_id, gaining_player_id}
        """
        self.turnovers = []
        
        if len(team_ball_control) < 2:
            return self.turnovers

        # Possession bölümleri oluştur (ardışık aynı takım frame'leri)
        possession_segments = []
        current_team = team_ball_control[0]
        segment_start = 0

        for i in range(1, len(team_ball_control)):
            if team_ball_control[i] != current_team:
                possession_segments.append({
                    "team": int(current_team),
                    "start_frame": segment_start,
                    "end_frame": i - 1,
                    "duration": i - segment_start
                })
                current_team = team_ball_control[i]
                segment_start = i

        # Son segmenti ekle
        possession_segments.append({
            "team": int(current_team),
            "start_frame": segment_start,
            "end_frame": len(team_ball_control) - 1,
            "duration": len(team_ball_control) - segment_start
        })

        # Min süreden kısa olanları filtrele (gürültü)
        valid_segments = [s for s in possession_segments 
                         if s["duration"] >= self.min_possession_frames]

        # Ardışık farklı takım segmentleri → turnover
        for i in range(1, len(valid_segments)):
            prev = valid_segments[i - 1]
            curr = valid_segments[i]

            if prev["team"] != curr["team"]:
                turnover_frame = curr["start_frame"]

                # Kaybeden oyuncuyu bul (prev segmentin son frame'inde topa sahip olan)
                losing_player_id = self._find_ball_holder(
                    tracks, prev["end_frame"])
                
                # Kazanan oyuncuyu bul (curr segmentin ilk frame'inde topa sahip olan)
                gaining_player_id = self._find_ball_holder(
                    tracks, curr["start_frame"])

                self.turnovers.append({
                    "frame": turnover_frame,
                    "losing_team": prev["team"],
                    "gaining_team": curr["team"],
                    "losing_player_id": losing_player_id,
                    "gaining_player_id": gaining_player_id,
                    "prev_possession_duration": prev["duration"],
                })

        print(f"⚽ Top Kaybı Analizi: {len(self.turnovers)} turnover tespit edildi")
        return self.turnovers

    def _find_ball_holder(self, tracks, frame_num):
        """Belirli bir frame'de topu tutan oyuncunun ID'sini bulur."""
        if frame_num >= len(tracks["players"]):
            return None
            
        for player_id, player_data in tracks["players"][frame_num].items():
            if player_data.get("has_ball", False):
                return player_id
        return None

    def get_summary(self):
        """Top kaybı istatistik özeti döndürür."""
        if not self.turnovers:
            return "Henüz analiz yapılmadı."

        team1_losses = sum(1 for t in self.turnovers if t["losing_team"] == 1)
        team2_losses = sum(1 for t in self.turnovers if t["losing_team"] == 2)

        summary = (
            f"═══ TOP KAYBI ÖZETİ ═══\n"
            f"Toplam Turnover: {len(self.turnovers)}\n"
            f"Takım 1 Top Kayıpları: {team1_losses}\n"
            f"Takım 2 Top Kayıpları: {team2_losses}\n"
            f"{'─' * 30}\n"
        )

        for i, t in enumerate(self.turnovers, 1):
            summary += (
                f"[{i}] Frame {t['frame']}: "
                f"Takım {t['losing_team']} → Takım {t['gaining_team']} "
                f"(Kaybeden: #{t['losing_player_id']}, "
                f"Kazanan: #{t['gaining_player_id']})\n"
            )

        return summary

    def draw_turnover_markers(self, frames, turnovers=None):
        """Frame'ler üzerine top kaybı anlarını işaretler."""
        import cv2

        if turnovers is None:
            turnovers = self.turnovers

        # Turnover frame'lerini set olarak tut (hızlı lookup)
        turnover_frames = {}
        for t in turnovers:
            # Turnover frame'i ve sonrasındaki 30 frame boyunca göster
            for f in range(t["frame"], min(t["frame"] + 30, len(frames))):
                turnover_frames[f] = t

        output_frames = []
        for frame_num, frame in enumerate(frames):
            if frame_num in turnover_frames:
                t = turnover_frames[frame_num]
                # Kırmızı banner
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 980), (600, 1080), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                cv2.putText(frame,
                            f"TURNOVER! Team {t['losing_team']} -> Team {t['gaining_team']}",
                            (20, 1030), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)
                cv2.putText(frame,
                            f"Lost by #{t['losing_player_id']} | Won by #{t['gaining_player_id']}",
                            (20, 1060), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 255), 2)

            output_frames.append(frame)

        return output_frames
