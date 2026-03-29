def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def get_bbox_width(bbox):
    return int(bbox[2] - bbox[0])

def measure_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def measure_xy_distance(point1, point2):
    """Returns signed (dx, dy) vector from point1 to point2."""
    return point2[0] - point1[0], point2[1] - point1[1]

def get_foot_position(bbox):
    """Returns the bottom-center point of a bounding box (player's feet)."""
    x1, _, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)