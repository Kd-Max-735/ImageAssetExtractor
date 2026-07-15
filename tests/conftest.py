from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw


def image_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    stream = BytesIO()
    image.save(stream, format=format)
    return stream.getvalue()


def transparent_sheet() -> Image.Image:
    image = Image.new("RGBA", (160, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 39, 39), fill=(220, 30, 30, 255))
    draw.ellipse((100, 50, 139, 89), fill=(30, 80, 220, 180))
    return image


def solid_sheet(background=(255, 255, 255), colored=False) -> Image.Image:
    image = Image.new("RGB", (180, 110), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 42, 42), fill=(20, 20, 20) if not colored else (220, 40, 40))
    draw.ellipse((110, 55, 150, 95), fill=(20, 20, 20) if not colored else (30, 90, 220))
    return image

