import cv2
from utils import measure_distance, get_foot_position


class SpeedAndDistanceEstimator:
    """
    Dönüştürülmüş saha koordinatları üzerinden oyuncuların
    hız (km/h) ve toplam koşu mesafesini (m) hesaplar.
    """

    def __init__(self, frame_window=5, frame_rate=24):
        self.frame_window = frame_window  # Kaç frame'de bir hesapla
        self.frame_rate = frame_rate

    def add_speed_and_distance_to_tracks(self, tracks):
        """Her oyuncuya hız ve kümülatif mesafe bilgisi ekler."""
        total_distance = {}

        for object_name, object_tracks in tracks.items():
            if object_name == "ball" or object_name == "referees":
                continue

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id].get('position_transformed')
                    end_position = object_tracks[last_frame][track_id].get('position_transformed')

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    if time_elapsed <= 0:
                        continue

                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    if object_name not in total_distance:
                        total_distance[object_name] = {}
                    if track_id not in total_distance[object_name]:
                        total_distance[object_name][track_id] = 0

                    total_distance[object_name][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object_name][frame_num_batch]:
                            continue
                        tracks[object_name][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object_name][frame_num_batch][track_id]['distance'] = \
                            total_distance[object_name][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        """Frame'ler üzerine hız ve mesafe bilgisini çizer."""
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object_name, object_tracks in tracks.items():
                if object_name == "ball" or object_name == "referees":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed')
                        distance = track_info.get('distance')
                        if speed is None or distance is None:
                            continue

                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed:.2f} km/h", position,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m",
                                    (position[0], position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            output_frames.append(frame)

        return output_frames
