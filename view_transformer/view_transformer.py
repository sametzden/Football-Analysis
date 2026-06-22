import numpy as np
import cv2
import json
import os


class ViewTransformer:
    """
    Piksel koordinatlarını gerçek saha koordinatlarına (metre) dönüştürür.

    İki mod destekler:
    1. SoccerNet Modu: GameState verisindeki pitch line etiketlerinden 
       homografi matrisi hesaplanır (frame başına dinamik).
    2. Sabit Mod (Fallback): Videoya özel sabit 4 nokta üzerinden
       perspektif dönüşümü yapılır.
    """

    # FIFA standart saha boyutları (metre)
    PITCH_LENGTH = 105.0  # x ekseni
    PITCH_WIDTH = 68.0    # y ekseni

    # SoccerNet pitch line tanımları → gerçek saha koordinatları (metre)
    # Orijin: saha sol-üst köşe (0,0)
    # x: 0..105, y: 0..68
    PITCH_LINE_COORDS = {
        "Side line left":       {"x": 0.0},
        "Side line right":      {"x": 105.0},
        "Side line top":        {"y": 0.0},
        "Side line bottom":     {"y": 68.0},
        "Middle line":          {"x": 52.5},
        "Big rect. left main":  {"x": 16.5},
        "Big rect. left top":   {"y": 13.84},
        "Big rect. left bottom": {"y": 54.16},
        "Big rect. right main": {"x": 88.5},
        "Big rect. right top":  {"y": 13.84},
        "Big rect. right bottom": {"y": 54.16},
        "Small rect. left main": {"x": 5.5},
        "Small rect. left top":  {"y": 24.84},
        "Small rect. left bottom": {"y": 43.16},
        "Small rect. right main": {"x": 99.5},
        "Small rect. right top":  {"y": 24.84},
        "Small rect. right bottom": {"y": 43.16},
    }

    def __init__(self, gamestate_json_path=None, pixel_vertices=None, target_vertices=None, frame_offset=0):
        """
        Args:
            gamestate_json_path: SoccerNet Labels-GameState.json yolu (dinamik mod)
            pixel_vertices: Sabit 4 piksel noktası (fallback mod)
            target_vertices: Sabit 4 gerçek saha noktası (fallback mod)
            frame_offset: Kesilmiş videolarda frame numarası kaydırması
        """
        self.frame_offset = frame_offset
        self.mode = "static"
        self.perspective_transformer = None
        self.pixel_vertices = None
        self.homography_per_frame = {}

        if gamestate_json_path and os.path.exists(gamestate_json_path):
            self._load_soccernet_data(gamestate_json_path)
            self.mode = "soccernet"
            print(f"📐 ViewTransformer: SoccerNet modu aktif ({len(self.pitch_annotations)} frame)")
        elif pixel_vertices is not None and target_vertices is not None:
            self.pixel_vertices = np.array(pixel_vertices, dtype=np.float32)
            target_vertices = np.array(target_vertices, dtype=np.float32)
            self.perspective_transformer = cv2.getPerspectiveTransform(
                self.pixel_vertices, target_vertices)
            self.mode = "static"
            print("📐 ViewTransformer: Sabit perspektif modu aktif")
        else:
            # Varsayılan sabit noktalar (08fd33_4.mp4 videosu için)
            court_width = 68
            court_length = 23.32
            self.pixel_vertices = np.array([
                [110, 1035],
                [265, 275],
                [910, 260],
                [1640, 915]
            ], dtype=np.float32)
            target_vertices = np.array([
                [0, court_width],
                [0, 0],
                [court_length, 0],
                [court_length, court_width]
            ], dtype=np.float32)
            self.perspective_transformer = cv2.getPerspectiveTransform(
                self.pixel_vertices, target_vertices)
            self.mode = "static"
            print("📐 ViewTransformer: Varsayılan sabit perspektif modu aktif")

    def _load_soccernet_data(self, json_path):
        """SoccerNet GameState verisinden pitch etiketlerini yükler."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.image_info = {}
        for img in data.get('images', []):
            self.image_info[img['image_id']] = img

        # Pitch annotations → frame başına pitch line noktaları
        self.pitch_annotations = {}
        for ann in data.get('annotations', []):
            if ann.get('supercategory') == 'pitch' and 'lines' in ann:
                image_id = ann['image_id']
                self.pitch_annotations[image_id] = ann['lines']

        # Object annotations → bbox_pitch bilgisi (doğrudan saha koordinatları)
        self.object_pitch_coords = {}
        for ann in data.get('annotations', []):
            if ann.get('supercategory') == 'object' and 'bbox_pitch' in ann:
                image_id = ann['image_id']
                if image_id not in self.object_pitch_coords:
                    self.object_pitch_coords[image_id] = []
                self.object_pitch_coords[image_id].append(ann)

        # Frame başına homografi matrisi hesapla
        self._compute_homographies()

    def _compute_homographies(self):
        """
        Her frame için pitch line noktalarından homografi matrisi hesaplar.
        Pitch line'ların bilinen gerçek saha koordinatları ile
        piksel koordinatları eşleştirilir.
        """
        for image_id, lines in self.pitch_annotations.items():
            img_info = self.image_info.get(image_id, {})
            img_w = img_info.get('width', 1920)
            img_h = img_info.get('height', 1080)

            pixel_points = []
            world_points = []

            for line_name, line_points in lines.items():
                if line_name not in self.PITCH_LINE_COORDS:
                    continue

                line_info = self.PITCH_LINE_COORDS[line_name]

                for pt in line_points:
                    px = pt['x'] * img_w
                    py = pt['y'] * img_h

                    # Çizginin yönüne göre saha koordinatını belirle
                    if "Side line top" == line_name:
                        # y=0 olan üst çizgi, x pozisyonunu tahmin etmemiz lazım
                        # normalize x'i saha uzunluğuna map et
                        world_x = pt['x'] * self.PITCH_LENGTH
                        world_y = 0.0
                    elif "Side line bottom" == line_name:
                        world_x = pt['x'] * self.PITCH_LENGTH
                        world_y = self.PITCH_WIDTH
                    elif "Side line left" == line_name:
                        world_x = 0.0
                        world_y = pt['y'] * self.PITCH_WIDTH
                    elif "Side line right" == line_name:
                        world_x = self.PITCH_LENGTH
                        world_y = pt['y'] * self.PITCH_WIDTH
                    elif "Middle line" == line_name:
                        world_x = 52.5
                        world_y = pt['y'] * self.PITCH_WIDTH
                    elif "x" in line_info:
                        world_x = line_info["x"]
                        world_y = pt['y'] * self.PITCH_WIDTH
                    elif "y" in line_info:
                        world_x = pt['x'] * self.PITCH_LENGTH
                        world_y = line_info["y"]
                    else:
                        continue

                    pixel_points.append([px, py])
                    world_points.append([world_x, world_y])

            if len(pixel_points) >= 4:
                pixel_pts = np.array(pixel_points, dtype=np.float32)
                world_pts = np.array(world_points, dtype=np.float32)

                H, mask = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    self.homography_per_frame[image_id] = H

    def transform_point(self, point, frame_num=None):
        """
        Tek bir piksel noktasını saha koordinatına dönüştürür.

        Args:
            point: (x, y) piksel koordinatı
            frame_num: Frame numarası (SoccerNet modu için)
        """
        if self.mode == "soccernet" and frame_num is not None:
            return self._transform_soccernet(point, frame_num)
        else:
            return self._transform_static(point)

    def _transform_static(self, point):
        """Sabit perspektif matrisiyle dönüşüm yapar."""
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transformed.reshape(-1, 2)

    def _transform_soccernet(self, point, frame_num):
        """SoccerNet homografi matrisiyle dönüşüm yapar."""
        # Frame numarasından image_id'ye dönüşüm
        # SoccerNet formatı: image_id tipik olarak "2021000001" gibi
        # Frame numarası 0-indexed, image_id için sahnedeki formatı kullan
        image_id = self._frame_to_image_id(frame_num)

        H = self.homography_per_frame.get(image_id)
        if H is None:
            # En yakın frame'in homografisini kullan
            H = self._get_nearest_homography(frame_num)
            if H is None:
                return None

        reshaped_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped_point, H)
        result = transformed.reshape(-1, 2)

        # Saha sınırları kontrolü (-5m tolerans)
        x, y = result[0]
        if -5 < x < self.PITCH_LENGTH + 5 and -5 < y < self.PITCH_WIDTH + 5:
            return result
        return None

    def _frame_to_image_id(self, frame_num):
        """Frame numarasından SoccerNet image_id'ye çevirir."""
        # image_id formatı: sahne prefix + 6 haneli frame no
        # Örnek: "2021000001" → sahne 021, frame 1
        if self.image_info:
            # Mevcut image_id'lerden deseni çıkar
            sample_id = list(self.image_info.keys())[0]
            prefix = sample_id[:-6]  # Son 6 hane frame numarası
            # frame_num (0'dan başlar), +1 (dosyalar 1'den başlar), + offset
            target_frame_num = frame_num + 1 + self.frame_offset
            return f"{prefix}{target_frame_num:06d}"
        return str(frame_num + self.frame_offset)

    def _get_nearest_homography(self, frame_num):
        """En yakın frame'in homografi matrisini bulur."""
        if not self.homography_per_frame:
            return None

        target_id = self._frame_to_image_id(frame_num)
        # Anahtar sıralı olarak en yakını bul
        keys = sorted(self.homography_per_frame.keys())
        if not keys:
            return None

        # Basit yaklaşım: ilk mevcut olanı kullan
        closest = min(keys, key=lambda k: abs(int(k[-6:]) - (frame_num + 1)))
        return self.homography_per_frame[closest]

    def add_transformed_position_to_tracks(self, tracks):
        """Tüm track'lere saha koordinatı bilgisini ekler."""
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    if 'position_adjusted' not in track_info:
                        continue
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position, frame_num)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object_name][frame_num][track_id]['position_transformed'] = position_transformed
