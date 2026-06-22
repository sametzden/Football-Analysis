import cv2
import numpy as np

class MiniMap:
    def __init__(self, pitch_length=105, pitch_width=68, scale=3.0, margin=20):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.scale = scale
        self.margin = margin
        self.width = int(pitch_length * scale) + 2 * margin
        self.height = int(pitch_width * scale) + 2 * margin
        self.bg_color = (0, 100, 0)
        self.line_color = (255, 255, 255)
        self.base_map = self._create_base_map()

    def _create_base_map(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.bg_color
        
        # Dış çizgiler
        cv2.rectangle(img, (self.margin, self.margin), 
                      (self.width - self.margin, self.height - self.margin), 
                      self.line_color, 2)
        # Orta çizgi
        mid_x = self.margin + int((self.pitch_length / 2) * self.scale)
        cv2.line(img, (mid_x, self.margin), (mid_x, self.height - self.margin), self.line_color, 2)
        
        # Orta yuvarlak
        mid_y = self.margin + int((self.pitch_width / 2) * self.scale)
        cv2.circle(img, (mid_x, mid_y), int(9.15 * self.scale), self.line_color, 2)
        
        # Sol ceza sahası
        cv2.rectangle(img, (self.margin, self.margin + int(13.84 * self.scale)), 
                      (self.margin + int(16.5 * self.scale), self.margin + int(54.16 * self.scale)), 
                      self.line_color, 2)
        
        # Sağ ceza sahası
        cv2.rectangle(img, (self.width - self.margin - int(16.5 * self.scale), self.margin + int(13.84 * self.scale)), 
                      (self.width - self.margin, self.margin + int(54.16 * self.scale)), 
                      self.line_color, 2)
                      
        return img

    def draw(self, frame, tracks, frame_num, team_assigner, actions=None):
        minimap = self.base_map.copy()
        
        for obj_name, obj_tracks in tracks.items():
            if frame_num >= len(obj_tracks):
                continue
            for track_id, track_info in obj_tracks[frame_num].items():
                pos = track_info.get('position_transformed')
                if pos is None:
                    continue
                    
                x_m, y_m = pos[0], pos[1]
                if x_m < -5 or x_m > self.pitch_length + 5 or y_m < -5 or y_m > self.pitch_width + 5:
                    continue

                # Taç çizgisi bug'ı: kenar değerleri clamp'le
                x_m = max(1.0, min(self.pitch_length - 1.0, x_m))
                y_m = max(1.0, min(self.pitch_width - 1.0, y_m))
                    
                x_px = self.margin + int(x_m * self.scale)
                y_px = self.margin + int(y_m * self.scale)
                
                if obj_name == 'players':
                    team = track_info.get('team')
                    color = team_assigner.team_colors.get(team, (255, 255, 255)) if team is not None else (128,128,128)
                    cv2.circle(minimap, (x_px, y_px), 5, color, -1)
                    cv2.circle(minimap, (x_px, y_px), 5, (0,0,0), 1)
                elif obj_name == 'goalkeepers':
                    team = track_info.get('team')
                    color = team_assigner.team_colors.get(team, (255, 255, 255)) if team is not None else (128,128,128)
                    cv2.circle(minimap, (x_px, y_px), 6, color, -1)
                    cv2.circle(minimap, (x_px, y_px), 6, (0,0,255), 2)
                elif obj_name == 'referees':
                    cv2.circle(minimap, (x_px, y_px), 4, (0, 255, 255), -1)
                elif obj_name == 'ball':
                    cv2.circle(minimap, (x_px, y_px), 4, (255, 255, 255), -1)
                    cv2.circle(minimap, (x_px, y_px), 6, (0, 0, 255), 2)

        # Aksiyon okları çiz (pas ve şut)
        if actions:
            self._draw_action_arrows(minimap, actions, frame_num)
                    
        # Minimap'i videonun sağ üst köşesine yerleştir
        h, w = minimap.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        
        # Saydamlık (alpha blending)
        alpha = 0.8
        if frame_w > w and frame_h > h:
            roi = frame[20:20+h, frame_w-w-20:frame_w-20]
            blended = cv2.addWeighted(roi, 1 - alpha, minimap, alpha, 0)
            frame[20:20+h, frame_w-w-20:frame_w-20] = blended
            
        return frame

    def _draw_action_arrows(self, minimap, actions, frame_num):
        """Minimap üzerinde aktif aksiyonları ok olarak çizer."""
        for action in actions:
            start_frame = action["frame"]
            end_frame = action.get("display_until", start_frame + 25)
            
            if not (start_frame <= frame_num <= end_frame):
                continue
                
            action_type = action["type"]
            if action_type == "DRİBBLİNG":
                continue  # Dribbling için ok çizmeye gerek yok
                
            ball_start = action.get("ball_start")
            ball_end = action.get("ball_end")
            
            if not ball_start or not ball_end:
                continue
                
            # Metre → piksel dönüşümü
            sx = self.margin + int(ball_start[0] * self.scale)
            sy = self.margin + int(ball_start[1] * self.scale)
            ex = self.margin + int(ball_end[0] * self.scale)
            ey = self.margin + int(ball_end[1] * self.scale)
            
            # Sınır kontrolü
            if not (0 < sx < self.width and 0 < sy < self.height):
                continue
            if not (0 < ex < self.width and 0 < ey < self.height):
                continue
            
            # Renk: PAS = mavi, ŞUT = kırmızı
            if action_type == "PAS":
                arrow_color = (255, 200, 0)  # Mavi-cyan
            elif action_type == "ŞUT":
                arrow_color = (0, 0, 255)    # Kırmızı
            else:
                arrow_color = (200, 200, 200)
            
            # Ok çiz
            cv2.arrowedLine(minimap, (sx, sy), (ex, ey), arrow_color, 2, tipLength=0.25)
            
            # Aksiyon etiketini ok başlangıcına yaz
            label = "P" if action_type == "PAS" else "Ş"
            cv2.putText(minimap, label, (sx - 5, sy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, arrow_color, 1)
