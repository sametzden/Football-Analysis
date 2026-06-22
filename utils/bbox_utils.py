def get_center_of_bbox(bbox):
    """Bounding box'ın merkez noktasını döndürür."""
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def get_bbox_width(bbox):
    """Bounding box genişliğini döndürür."""
    return int(bbox[2] - bbox[0])


def measure_distance(point1, point2):
    """İki nokta arasındaki Öklid mesafesini döndürür."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def measure_xy_distance(point1, point2):
    """İki nokta arasındaki (dx, dy) vektörünü döndürür."""
    return point2[0] - point1[0], point2[1] - point1[1]


def get_foot_position(bbox):
    """Bounding box'ın alt-orta noktasını (oyuncunun ayakları) döndürür."""
    x1, _, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
