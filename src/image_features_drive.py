import gc
import warnings
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageStat
from concurrent.futures import ThreadPoolExecutor
import requests
from io import BytesIO
import threading

warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

import random
np.random.seed(42)
random.seed(42)

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

IMAGE_SIZE = (224, 224)
_MAX_CACHE  = 10_000          # cap to avoid unbounded RAM growth
_bytes_cache: dict[str, bytes] = {}
_cache_lock  = threading.Lock()

# ── Image fetching ────────────────────────────────────────────────────────────

def _fetch_bytes(url: str) -> bytes | None:
    """Download image bytes and cache them (compressed = small RAM footprint)."""
    if not url or not isinstance(url, str):
        return None
    with _cache_lock:
        if url in _bytes_cache:
            return _bytes_cache[url]
    try:
        resp = requests.get(url, timeout=8)
        data = resp.content
        with _cache_lock:
            if len(_bytes_cache) < _MAX_CACHE:
                _bytes_cache[url] = data
        return data
    except Exception:
        return None


def _fetch_image(url: str) -> Image.Image | None:
    """Return decoded PIL image from bytes cache (decode on-demand, not stored)."""
    data = _fetch_bytes(url)
    if data is None:
        return None
    try:
        with BytesIO(data) as buf:
            return Image.open(buf).convert('RGB').copy()  # .copy() forces full load before buf closes
    except Exception:
        return None


def _prefetch_images(links: list[str], max_workers: int = 32):
    """Parallel prefetch of compressed bytes — memory-safe."""
    unique = [l for l in links if l and l not in _bytes_cache]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(tqdm(ex.map(_fetch_bytes, unique),
                  total=len(unique), desc="⬇️  Downloading images", leave=False))


# ── Traditional feature extraction (single image load per sample) ─────────────

def _extract_traditional(img: Image.Image) -> dict:
    """Extract lightweight traditional features from a pre-loaded PIL image."""
    zero = {
        'brightness': 0, 'contrast': 0, 'color_variance': 0,
        'dominant_r': 0, 'dominant_g': 0, 'dominant_b': 0,
        'texture_variance': 0, 'aspect_ratio': 1.0,
        'center_brightness': 0, 'image_complexity': 0,
    }
    if img is None:
        return zero
    try:
        stat     = ImageStat.Stat(img)
        arr      = np.array(img)
        small    = np.array(img.resize((32, 32))).reshape(-1, 3).astype(np.float32)
        gray     = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w     = arr.shape[:2]
        ch, cw   = h // 4, w // 4
        center   = arr[ch:3*ch, cw:3*cw]
        dominant = small.mean(axis=0)

        return {
            'brightness':        float(np.mean(stat.mean)),
            'contrast':          float(np.std(stat.mean)),
            'color_variance':    float(np.var(small, axis=0).mean()),
            'dominant_r':        float(dominant[0]),
            'dominant_g':        float(dominant[1]),
            'dominant_b':        float(dominant[2]),
            'texture_variance':  float(np.var(gray)),       # dropped: edge_density (Canny is slow)
            'aspect_ratio':      float(w / h),
            'center_brightness': float(np.mean(center)),
            'image_complexity':  float(np.std(arr)),        # dropped: color_richness (np.unique is slow)
        }
    except Exception:
        return zero


# ── MobileNetV3-Small deep features (576-dim, ~2.5MB, fast on CPU) ───────────

_mobilenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
]) if TORCH_AVAILABLE else None


def _load_mobilenet(device: str) -> nn.Module:
    """MobileNetV3-Small feature extractor (576-dim, ~2.5MB weights)."""
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    m.classifier = nn.Identity()   # strip classifier → raw 576-dim embeddings
    return m.eval().to(device)


def extract_deep_features_mobilenet(links: list[str], batch_size: int = 256) -> np.ndarray:
    """
    MobileNetV3-Small embeddings (576-dim).
    Lighter than DINOv2: ~2.5MB weights, larger batches, faster on CPU.
    """
    if not TORCH_AVAILABLE:
        return np.zeros((len(links), 576), dtype=np.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # MPS skipped — segfault on macOS
    print(f"   MobileNetV3 device: {device}")
    model = _load_mobilenet(device)

    all_feats = []
    for i in tqdm(range(0, len(links), batch_size), desc="🔬 MobileNet batches"):
        batch_links = links[i:i + batch_size]
        tensors = []
        for url in batch_links:
            img = _fetch_image(url)
            tensors.append(_mobilenet_transform(img) if img is not None else torch.zeros(3, 224, 224))

        batch_tensor = torch.stack(tensors).to(device)
        with torch.inference_mode():           # lighter than no_grad: disables autograd entirely
            feats = model(batch_tensor).cpu().numpy().astype(np.float32)
        all_feats.append(feats)

        del batch_tensor, tensors
        gc.collect()

    del model
    gc.collect()
    return np.vstack(all_feats)


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_comprehensive_image_features_drive(
    df: pd.DataFrame,
    use_deep_features: bool = True,
    model_name: str = 'mobilenet',       # ignored, always MobileNetV3-Small
    credentials_path: str = 'credentials.json',
    folder_id: str = '1NbCCpQBHPAsZXmqa1p5f8l_kYvjmG3hI',
    prefetch_workers: int = 32,
    batch_size: int = 256,               # larger batches since model is lighter
) -> tuple[np.ndarray, list[str]]:
    """
    Lightweight image feature extraction.
    - Downloads each image ONCE via parallel HTTP
    - Traditional features: 10 features, single pass (no Canny / np.unique)
    - Deep features: MobileNetV3-Small (576-dim, ~2.5MB, fast on CPU)
    """
    links = df['image_link'].tolist()

    print("⬇️  Prefetching images in parallel...")
    _prefetch_images(links, max_workers=prefetch_workers)

    print("🎨 Extracting traditional features...")
    trad_rows = [_extract_traditional(_fetch_image(l)) for l in tqdm(links, leave=False)]
    traditional_features = np.array([list(r.values()) for r in trad_rows], dtype=np.float32)
    trad_names = list(trad_rows[0].keys())

    if use_deep_features:
        print("🔬 Extracting MobileNetV3-Small deep features...")
        deep_features  = extract_deep_features_mobilenet(links, batch_size=batch_size)
        final_features = np.hstack([traditional_features, deep_features])
        feature_names  = trad_names + [f'mobilenet_{i}' for i in range(deep_features.shape[1])]
    else:
        final_features = traditional_features
        feature_names  = trad_names

    print(f"✅ Image features ready — shape: {final_features.shape}")
    return final_features, feature_names
