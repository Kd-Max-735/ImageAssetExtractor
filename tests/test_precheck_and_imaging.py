from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from image_asset_extractor.errors import ExtractorError
from image_asset_extractor.imaging.background import classify_background
from image_asset_extractor.imaging.components import analyze_components
from image_asset_extractor.imaging.foreground import segment_foreground
from image_asset_extractor.imaging.grid import projection_cells
from image_asset_extractor.imaging.morphology import adaptive_kernel, clean_detection_mask
from image_asset_extractor.imaging.precheck import load_and_precheck
from image_asset_extractor.imaging.splitting import projection_split
from image_asset_extractor.schemas.config import ExtractionConfig

from conftest import image_bytes, solid_sheet, transparent_sheet


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
def test_supported_formats(format):
    image = solid_sheet()
    checked = load_and_precheck(image_bytes(image, format), ExtractionConfig())
    assert checked.source_format == format
    assert checked.width == 180


def test_transparent_background_and_soft_alpha():
    checked = load_and_precheck(image_bytes(transparent_sheet()), ExtractionConfig())
    background = classify_background(checked)
    detection, alpha = segment_foreground(checked, background, ExtractionConfig())
    assert background.type == "transparent"
    assert set(np.unique(alpha)) >= {0, 180, 255}
    assert np.count_nonzero(detection) > 0


def test_legacy_shadow_modes_preserve_appearance_and_text_mode_still_applies():
    image = Image.new("RGBA", (120, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 15, 70, 55), fill=(40, 80, 180, 255))
    draw.ellipse((35, 52, 85, 68), fill=(20, 20, 20, 60))
    draw.rectangle((100, 8, 102, 10), fill=(20, 20, 20, 255))
    checked = load_and_precheck(image_bytes(image), ExtractionConfig())
    background = classify_background(checked)
    icon_detection, icon_alpha = segment_foreground(checked, background, ExtractionConfig(asset_mode="icon", shadow_mode="auto"))
    product_detection, product_alpha = segment_foreground(checked, background, ExtractionConfig(asset_mode="product", shadow_mode="auto"))
    text_removed, _ = segment_foreground(checked, background, ExtractionConfig(asset_mode="product", shadow_mode="auto", text_mode="remove"))
    assert np.array_equal(product_alpha, icon_alpha)
    assert np.array_equal(product_detection, icon_detection)
    assert product_alpha[62, 80] > 0
    assert text_removed[10, 101] == 0


def test_legacy_remove_shadow_value_is_normalized_to_preserve():
    image = Image.new("RGBA", (120, 90), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 25, 90, 75), fill=(20, 220, 240, 70))
    draw.ellipse((58, 50, 105, 78), fill=(35, 35, 35, 65))
    draw.rectangle((48, 28, 82, 62), fill=(40, 90, 220, 255))
    checked = load_and_precheck(image_bytes(image), ExtractionConfig())
    background = classify_background(checked)
    config = ExtractionConfig(asset_mode="icon", shadow_mode="remove")
    config.validate()
    detection, alpha = segment_foreground(checked, background, config)
    preserved_detection, preserved_alpha = segment_foreground(
        checked, background, ExtractionConfig(asset_mode="icon", shadow_mode="preserve")
    )
    assert config.shadow_mode == "preserve"
    assert alpha[70, 98] > 0
    assert alpha[45, 42] > 0
    assert np.array_equal(detection, preserved_detection)
    assert np.array_equal(alpha, preserved_alpha)


def test_component_shadow_feature_is_not_hardcoded_false():
    rgba = np.zeros((30, 30, 4), np.uint8)
    rgba[10:20, 8:24] = (40, 42, 41, 70)
    regions, _ = analyze_components((rgba[:, :, 3] > 0).astype(np.uint8) * 255, rgba, 1)
    assert regions[0].suspected_shadow


@pytest.mark.parametrize("background", [(255, 255, 255), (210, 220, 230), (40, 120, 70)])
def test_solid_background_segmentation(background):
    checked = load_and_precheck(image_bytes(solid_sheet(background, colored=True)), ExtractionConfig())
    info = classify_background(checked)
    detection, alpha = segment_foreground(checked, info, ExtractionConfig())
    assert info.type == "solid"
    assert detection[20, 20] == 255
    assert detection[0, 0] == 0
    assert alpha[20, 20] > alpha[0, 0]


def test_edge_foreground_does_not_replace_multi_edge_background():
    image = Image.new("RGB", (180, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 15, 45, 70), fill=(220, 50, 40))
    draw.ellipse((110, 0, 165, 50), fill=(30, 120, 210))
    checked = load_and_precheck(image_bytes(image), ExtractionConfig())
    info = classify_background(checked)
    detection, _ = segment_foreground(checked, info, ExtractionConfig())
    assert min(info.color) > 245
    assert np.count_nonzero(detection) / detection.size < 0.45
    assert detection[120, 90] == 0


def test_low_contrast_watermark_is_stable_but_jpeg_noise_is_not_promoted():
    watermark = Image.new("RGB", (180, 90), (250, 250, 250))
    ImageDraw.Draw(watermark).text((35, 35), "WATERMARK", fill=(241, 241, 241))
    checked = load_and_precheck(image_bytes(watermark), ExtractionConfig())
    info = classify_background(checked)
    detected, _ = segment_foreground(checked, info, ExtractionConfig())
    rng = np.random.default_rng(11)
    noise = np.clip(250 + rng.integers(-2, 3, size=(90, 180, 1)), 0, 255).astype(np.uint8)
    noise = np.repeat(noise, 3, axis=2)
    noisy = load_and_precheck(image_bytes(Image.fromarray(noise, "RGB"), "JPEG"), ExtractionConfig())
    noisy_info = classify_background(noisy)
    noisy_detection, _ = segment_foreground(noisy, noisy_info, ExtractionConfig())
    assert np.count_nonzero(detected) > 20
    assert np.count_nonzero(noisy_detection) < detected.size * 0.01


def test_components_expose_required_features():
    image = np.zeros((80, 100, 4), dtype=np.uint8)
    image[10:30, 10:30] = (255, 0, 0, 255)
    image[40:70, 60:90] = (0, 255, 0, 255)
    mask = (image[:, :, 3] > 0).astype(np.uint8) * 255
    regions, labels = analyze_components(mask, image, 5)
    assert len(regions) == 2
    assert all(r.area and r.perimeter and r.aspect_ratio for r in regions)
    assert all(len(r.average_color) == 3 and r.min_distance > 0 for r in regions)


def test_grid_projection_and_split_helpers():
    mask = np.zeros((60, 90), np.uint8)
    mask[5:20, 5:20] = 255
    mask[5:20, 40:55] = 255
    mask[35:50, 5:20] = 255
    mask[35:50, 40:55] = 255
    assert len(projection_cells(mask)) == 4
    assert len(projection_split(mask[5:20, 5:55])) == 2


def test_adaptive_morphology_kernel_changes_with_image_size():
    assert adaptive_kernel((100, 100), 0.01) < adaptive_kernel((1000, 1000), 0.01)


def test_thin_strokes_survive_cleanup_and_small_components_reach_ownership():
    mask = np.zeros((500, 500), np.uint8)
    mask[220:280, 220:280] = 255
    mask[200:215, 248:251] = 255
    rgba = np.zeros((500, 500, 4), np.uint8)
    rgba[mask > 0] = (240, 160, 20, 255)
    cleaned = clean_detection_mask(mask)
    regions, _ = analyze_components(cleaned, rgba, min_area=50)
    assert np.count_nonzero(cleaned[200:215, 248:251]) == 45
    assert any(region.area < 50 and region.suspected_noise for region in regions)


@pytest.mark.parametrize("payload,code", [(b"not an image", "IMAGE_DECODE_FAILED")])
def test_invalid_format(payload, code):
    with pytest.raises(ExtractorError) as error:
        load_and_precheck(payload, ExtractionConfig())
    assert error.value.code == code


def test_empty_and_fully_transparent_images():
    with pytest.raises(ExtractorError, match="no distinguishable"):
        load_and_precheck(image_bytes(Image.new("RGB", (20, 20), "white")), ExtractionConfig())
    with pytest.raises(ExtractorError) as error:
        load_and_precheck(image_bytes(Image.new("RGBA", (20, 20), (0, 0, 0, 0))), ExtractionConfig())
    assert error.value.code == "FULLY_TRANSPARENT"


def test_resource_limit():
    config = ExtractionConfig(max_dimension=10)
    with pytest.raises(ExtractorError) as error:
        load_and_precheck(image_bytes(solid_sheet()), config)
    assert error.value.code == "IMAGE_TOO_LARGE"


def test_exif_orientation_is_applied():
    image = Image.new("RGB", (40, 20), "white")
    ImageDraw.Draw(image).rectangle((2, 2, 10, 10), fill="black")
    exif = Image.Exif()
    exif[274] = 6
    from io import BytesIO
    stream = BytesIO()
    image.save(stream, "JPEG", exif=exif)
    checked = load_and_precheck(stream.getvalue(), ExtractionConfig())
    assert (checked.width, checked.height) == (20, 40)
