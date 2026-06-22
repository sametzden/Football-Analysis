from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    """
    Her frame'de topa en yakın oyuncuyu belirler.
    Oyuncunun ayak pozisyonu (bbox alt köşeleri) ile top merkezi arasındaki
    mesafeye göre atama yapar.
    """

    def __init__(self, max_distance=70):
        self.max_player_ball_distance = max_distance

    def assign_ball_to_players(self, players, ball_bbox):
        """
        Topa en yakın oyuncunun ID'sini döndürür.
        Eğer hiçbir oyuncu yeterince yakın değilse -1 döndürür.
        """
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = float('inf')
        assigned_player_id = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]

            # Oyuncunun sol ve sağ ayak noktalarından mesafeyi hesapla
            distance_left = measure_distance(
                (player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player_id = player_id

        return assigned_player_id
