"""Image decoding and geometric normalisation shared by every data source.

All sources (real A, external B, synthetic) go through `load_gray` + `to_square`
so that the classifier never sees source-specific preprocessing differences
introduced by us.
"""

from pathlib import Path

import numpy as np
from PIL import Image

DICOM_SUFFIXES = {".dcm", ".dicom"}


def load_gray(path: str | Path) -> np.ndarray:
    """Decode an image file (JPEG/PNG/DICOM) into a 2D uint8 array."""
    path = Path(path)
    if path.suffix.lower() in DICOM_SUFFIXES:
        return _load_dicom(path)
    with Image.open(path) as im:
        if im.mode in ("I", "I;16", "I;16B", "F"):
            return _to_uint8(np.asarray(im, dtype=np.float32))
        return np.asarray(im.convert("L"))


def _load_dicom(path: Path) -> np.ndarray:
    import pydicom

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    if arr.ndim == 3:  # RGB-encoded DICOM
        arr = arr.mean(axis=-1)
    return _to_uint8(arr)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).round().astype(np.uint8)


def to_square(img: np.ndarray, size: int, mode: str = "squash") -> np.ndarray:
    """Map an arbitrary-aspect image to `size` x `size`.

    squash    - resize the full field of view to a square (anatomy kept, aspect changed).
    letterbox - keep aspect ratio and zero-pad. In the Kermany data aspect ratio
                differs by class, so padding width leaks the label; use only for ablation.
    harmonize[F] - squash, then remove image-quality cues: area-downsample to a common resolution
                floor F (default 128; also averages out noise), equalise the histogram, resize back.
                In Kermany (and the synthetic data) softness and low contrast predict pneumonia.
    """
    pil = Image.fromarray(img)
    if mode == "squash":
        return np.asarray(pil.resize((size, size), Image.BICUBIC))
    if mode.startswith("harmonize"):
        floor = int(mode[len("harmonize"):] or 128)
        small = np.asarray(pil.resize((floor, floor), Image.BOX))
        return np.asarray(Image.fromarray(equalize(small)).resize((size, size), Image.BICUBIC))
    if mode == "letterbox":
        w, h = pil.size
        scale = size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        canvas = Image.new("L", (size, size), 0)
        canvas.paste(pil.resize((nw, nh), Image.BICUBIC), ((size - nw) // 2, (size - nh) // 2))
        return np.asarray(canvas)
    raise ValueError(f"unknown resize mode: {mode}")


def equalize(img: np.ndarray) -> np.ndarray:
    """Global histogram equalisation of a uint8 image (maps intensities to their CDF)."""
    hist = np.bincount(img.ravel(), minlength=256)
    cdf = hist.cumsum().astype(np.float64)
    cdf = (cdf - cdf[hist.nonzero()[0][0]]) / max(cdf[-1] - cdf[hist.nonzero()[0][0]], 1)
    return (cdf.clip(0, 1) * 255).round().astype(np.uint8)[img]


def image_stats(path: str | Path) -> dict:
    """Per-file properties used in EDA, plus a thumbnail embedding for duplicate detection."""
    path = Path(path)
    with Image.open(path) as im:
        mode = im.mode
    img = load_gray(path)
    h, w = img.shape
    return {"width": w, "height": h, "mode": mode, "file_bytes": path.stat().st_size,
            "mean_intensity": float(img.mean()), "std_intensity": float(img.std()),
            "thumb": thumbnail(img)}


def thumbnail(img: np.ndarray, size: int = 32) -> np.ndarray:
    """Zero-mean, unit-norm size x size thumbnail; cosine similarity > ~0.99 flags near-duplicates.

    (64-bit perceptual hashes are not discriminative enough for chest X-rays: thousands of
    distinct patients collide.)
    """
    t = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)
