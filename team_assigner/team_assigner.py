from sklearn.cluster import KMeans


class TeamAssigner:
    """
    Forma rengine göre oyuncuları 2 takıma ayırır.
    KMeans kümeleme kullanarak forma bölgesindeki baskın rengi çıkarır.
    """

    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.goalkeeper_team_dict = {}

    def get_clustering_model(self, image):
        """Görüntüyü 2 kümeye ayıran KMeans modeli döndürür."""
        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Oyuncunun bbox bölgesinin üst yarısından forma rengini çıkarır.
        Köşe pikselleri arka plan olarak kabul edilir.
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Üst yarı → forma bölgesi (bacaklar hariç)
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Kümeleme modelini oluştur
        kmeans = self.get_clustering_model(top_half_image)

        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Köşeler arka plan kümesini belirler
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        İlk frame'deki oyuncuların forma renklerini kullanarak
        2 takım rengini belirler.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """Bir oyuncunun hangi takıma ait olduğunu belirler (1 veya 2)."""
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # 0-indexed → 1-indexed

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_goalkeeper_team(self, frame, goalkeeper_bbox, goalkeeper_id, frame_width):
        """
        Kalecileri konum bazlı takıma atar.
        Sol yarı → Takım 1, Sağ yarı → Takım 2.
        """
        if goalkeeper_id in self.goalkeeper_team_dict:
            return self.goalkeeper_team_dict[goalkeeper_id]

        x1, _, x2, _ = goalkeeper_bbox
        x_center = (x1 + x2) / 2

        team_id = 1 if x_center < frame_width / 2 else 2
        self.goalkeeper_team_dict[goalkeeper_id] = team_id
        return team_id
