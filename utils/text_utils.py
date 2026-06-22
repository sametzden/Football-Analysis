"""
OpenCV Türkçe Karakter Desteği
================================
OpenCV'nin putText() fonksiyonu Türkçe karakterleri (ş, ğ, ü, ö, ı, ç)
desteklemez. Bu modül PIL/Pillow kullanarak Türkçe metni frame'e çizer.
"""

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Sistem fontlarını dene (bold, normal)
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _get_font(size):
    if not PIL_AVAILABLE:
        return None
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def put_text_tr(frame, text, position, font_size=22, color=(255, 255, 255),
                bg_color=None, bg_padding=4):
    """
    Türkçe karakter destekli metin çizer.

    Args:
        frame:       OpenCV BGR frame
        text:        Yazılacak metin (Türkçe karakterler dahil)
        position:    (x, y) sol üst köşe
        font_size:   Punto büyüklüğü
        color:       BGR renk (OpenCV sırası)
        bg_color:    Arka plan rengi BGR (None ise arka plan yok)
        bg_padding:  Arka plan için kenarlık piksel sayısı
    Returns:
        Metin eklenmiş frame
    """
    if not PIL_AVAILABLE:
        # Fallback: Türkçe karakterleri ASCII karşılığıyla yaz
        ascii_text = (text
                      .replace('ş', 's').replace('Ş', 'S')
                      .replace('ğ', 'g').replace('Ğ', 'G')
                      .replace('ü', 'u').replace('Ü', 'U')
                      .replace('ö', 'o').replace('Ö', 'O')
                      .replace('ı', 'i').replace('İ', 'I')
                      .replace('ç', 'c').replace('Ç', 'C'))
        rgb = (color[2], color[1], color[0])
        cv2.putText(frame, ascii_text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 30, rgb, 2)
        return frame

    # PIL ile çiz
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)

    x, y = position
    # PIL rengi RGB sırasında
    pil_color = (color[2], color[1], color[0])

    # Arka plan
    if bg_color is not None:
        bbox = draw.textbbox((x, y), text, font=font)
        bx1 = bbox[0] - bg_padding
        by1 = bbox[1] - bg_padding
        bx2 = bbox[2] + bg_padding
        by2 = bbox[3] + bg_padding
        pil_bg = (bg_color[2], bg_color[1], bg_color[0])
        draw.rectangle([bx1, by1, bx2, by2], fill=pil_bg)

    draw.text((x, y), text, font=font, fill=pil_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
