"""
CropCare AI - Image Preprocessing Module
Handles image validation, color space normalization, aspect ratio handling,
and model-specific tensor preprocessing for MobileNetV2.
"""

from typing import Tuple, Union, Optional
from PIL import Image
import numpy as np
import io
import cv2


def load_and_validate_image(
    source: Union[bytes, io.BytesIO, str, Image.Image, np.ndarray]
) -> Image.Image:
    """
    Loads an image from various input sources and validates its integrity.
    Always returns a clean RGB PIL.Image.
    """
    if source is None:
        raise ValueError("No image source provided.")

    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (str, bytes, io.BytesIO)):
        try:
            if isinstance(source, bytes):
                source = io.BytesIO(source)
            img = Image.open(source)
            # Verify image format can be decoded
            img.verify()
            # Reopen because verify() closes/invalidates the file pointer
            if hasattr(source, "seek"):
                source.seek(0)
            img = Image.open(source)
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image format: {e}")
    elif isinstance(source, np.ndarray):
        # Handle OpenCV BGR or Grayscale or RGB array
        if source.ndim == 2:
            img = Image.fromarray(source).convert("RGB")
        elif source.ndim == 3:
            if source.shape[2] == 3:
                # If originating from OpenCV cv2.imread or cv2 decode, it's BGR
                rgb_arr = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_arr)
            elif source.shape[2] == 4:
                rgb_arr = cv2.cvtColor(source, cv2.COLOR_BGRA2RGBA)
                img = Image.fromarray(rgb_arr).convert("RGB")
            else:
                raise ValueError(f"Unsupported numpy image channels: {source.shape[2]}")
        else:
            raise ValueError(f"Unsupported numpy image array shape: {source.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(source)}")

    # Ensure RGB color space
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def preprocess_for_inference(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    method: str = "mobilenet_v2"
) -> Tuple[np.ndarray, Image.Image]:
    """
    Preprocesses a PIL Image for MobileNetV2 deep learning inference.

    Args:
        image: Validated PIL Image in RGB mode.
        target_size: Tuple (width, height) expected by model. Default (224, 224).
        method: Preprocessing strategy ('mobilenet_v2', 'rescale_255', 'raw').

    Returns:
        Tuple of (batch_tensor, display_image):
            - batch_tensor: float32 ndarray with shape (1, height, width, 3)
            - display_image: PIL Image suitable for UI display
    """
    if not isinstance(image, Image.Image):
        image = load_and_validate_image(image)

    # Resize to exact model dimensions using high-quality bilinear interpolation
    resized_img = image.resize(target_size, resample=Image.Resampling.BILINEAR)

    # Convert to float32 NumPy array
    arr = np.array(resized_img, dtype=np.float32)

    # Apply selected normalization
    norm_method = (method or "mobilenet_v2").lower()
    if norm_method == "mobilenet_v2":
        # MobileNetV2 standard normalization: scales pixels from [0, 255] to [-1.0, 1.0]
        tensor = (arr / 127.5) - 1.0
    elif norm_method == "rescale_255":
        # Standard rescale to [0.0, 1.0]
        tensor = arr / 255.0
    elif norm_method == "raw":
        # Unscaled raw pixel values [0, 255]
        tensor = arr
    else:
        # Fallback to MobileNetV2 standard
        tensor = (arr / 127.5) - 1.0

    # Expand batch dimension: (height, width, 3) -> (1, height, width, 3)
    batch_tensor = np.expand_dims(tensor, axis=0)

    return batch_tensor, resized_img


def extract_image_metadata(image: Image.Image) -> dict:
    """
    Extracts basic diagnostics for the image to display in the UI.
    """
    return {
        "width": image.width,
        "height": image.height,
        "format": getattr(image, "format", "JPEG") or "JPEG",
        "mode": image.mode,
        "aspect_ratio": round(image.width / max(image.height, 1), 2)
    }


def validate_leaf_foliage(image: Image.Image) -> Tuple[bool, str, dict]:
    """
    Analyzes visual, chromatic, and foliar characteristics to verify if an image is plausibly
    a genuine crop leaf rather than a random non-plant object (e.g. circuit board, blue mat,
    face, animal, vehicle, furniture, blank graphic, text).

    Returns:
        Tuple of (is_valid: bool, status_message: str, diagnostics: dict)
    """
    from config import (
        MIN_FOLIAGE_COVERAGE,
        MIN_GREEN_TISSUE_RATIO,
        MIN_IMAGE_VARIANCE,
        MAX_UNNATURAL_COLOR_RATIO
    )

    if not isinstance(image, Image.Image):
        image = load_and_validate_image(image)

    arr = np.array(image.convert("RGB"))
    total_pixels = arr.shape[0] * arr.shape[1]

    # 1. Variance / Contrast Check (detects solid blanks, monochrome fields, flat artificial shapes)
    std_dev = float(np.std(arr))
    if std_dev < MIN_IMAGE_VARIANCE:
        return False, "The uploaded image lacks visual texture and variation (appears blank or uniform).", {
            "std_dev": round(std_dev, 2)
        }

    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)

    # 2. Unnatural Non-Plant Colors (Blues, Cyans, Magentas, Violets)
    # Natural agricultural leaves do not contain prominent blue or magenta hues.
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    blue_dominant = (b > 50) & (b > g * 1.12) & (b > r * 1.12)
    hsv_unnatural = (h >= 95) & (h <= 170) & (s >= 35) & (v >= 35)
    unnatural_mask = blue_dominant | hsv_unnatural
    unnatural_ratio = float(np.count_nonzero(unnatural_mask) / total_pixels)

    if unnatural_ratio > MAX_UNNATURAL_COLOR_RATIO:
        return False, (
            f"Prominent non-plant artificial colors detected ({unnatural_ratio:.1%} "
            f"blue/cyan/purple background, max allowed: {MAX_UNNATURAL_COLOR_RATIO:.0%}). "
            "Image appears to be a non-crop object."
        ), {
            "unnatural_ratio": round(unnatural_ratio * 100.0, 1),
            "max_allowed": round(MAX_UNNATURAL_COLOR_RATIO * 100.0, 1)
        }

    # 3. Foliar Vegetation & Chlorophyll Analysis
    # Excess Green Index: 2G - R - B
    exg = 2.0 * g - r - b

    # Chlorophyll green tissue: ExG > 12 and G > R and G > B with natural HSV leaf green
    green_tissue_mask = (exg > 12) & (g > r) & (g > b) & (h >= 25) & (h <= 95) & (s >= 25) & (v >= 25)
    green_ratio = float(np.count_nonzero(green_tissue_mask) / total_pixels)

    # Foliar chlorotic/necrotic lesion tissue (brown, yellow, tan blight/rust spots):
    # Require G > B + 12 and warm hues [10, 25] to distinguish leaf lesions from human skin or cardboard
    lesion_mask = (h >= 10) & (h <= 25) & (s >= 35) & (v >= 35) & (g > b + 12) & (r > b + 15)

    # Combined foliar mask (green chlorophyll + leaf disease lesions)
    foliar_mask = green_tissue_mask | lesion_mask
    foliar_ratio = float(np.count_nonzero(foliar_mask) / total_pixels)

    diagnostics = {
        "foliar_ratio": round(foliar_ratio * 100.0, 1),
        "green_ratio": round(green_ratio * 100.0, 1),
        "unnatural_ratio": round(unnatural_ratio * 100.0, 1),
        "std_dev": round(std_dev, 2)
    }

    # Foliar coverage requirement
    if foliar_ratio < MIN_FOLIAGE_COVERAGE:
        return False, (
            f"Insufficient crop foliage detected (foliar coverage: {foliar_ratio:.1%}, "
            f"minimum expected: {MIN_FOLIAGE_COVERAGE:.0%}). "
            "The uploaded photo does not appear to be a crop leaf."
        ), diagnostics

    # Chlorophyll presence requirement (real leaf must possess genuine plant tissue)
    if green_ratio < MIN_GREEN_TISSUE_RATIO:
        return False, (
            f"Lack of natural plant chlorophyll/vegetation (green leaf tissue: {green_ratio:.1%}, "
            f"minimum expected: {MIN_GREEN_TISSUE_RATIO:.0%}). "
            "The image does not show recognizable crop foliage."
        ), diagnostics

    return True, f"Valid crop foliage verified ({foliar_ratio:.1%} foliar coverage, {green_ratio:.1%} green tissue).", diagnostics
