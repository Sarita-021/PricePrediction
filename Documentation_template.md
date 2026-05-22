# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Vani
**Team Members:** Sarita, Gargi
**Solution Type:** Multi-Modal Machine Learning (Text + Image)
**Primary Model:** LightGBM Regression with Feature Fusion

---

## 1. Executive Summary

This solution addresses the Smart Product Pricing challenge through a multi-modal machine learning approach that combines advanced text processing and lightweight image analysis. Our system processes 50,000 training samples with catalog content and product images to predict optimal pricing using a LightGBM regression model.

**Key Achievements:**
- Shared feature extraction pipeline used across all execution modes
- Feature caching to disk — extract once, reuse forever across runs
- Lightweight image model (MobileNetV3-Small, ~2.5MB) replacing heavy DINOv2/ResNet50
- Separate commands for text extraction, image extraction, training, and prediction
- SMAPE scores ranging from 12-22% depending on execution mode and features used

---

## 2. Methodology Overview

### 2.1 Problem Analysis

**Challenge:** Predict product prices from heterogeneous data (text + images) with complex relationships between product attributes and pricing.

**Key Insights:**
- Brand recognition significantly impacts pricing
- Product categories have distinct pricing patterns
- Visual features (color, texture, composition) correlate with price ranges
- Text semantic clustering reveals premium vs budget product segments
- Item Pack Quantity (IPQ) directly influences unit pricing

### 2.2 Solution Strategy

**Multi-Modal Approach:**
1. **Text Processing:** Extract semantic features, brand information, and pricing indicators — shared across all pipelines via `src/feature_extraction.py`
2. **Image Analysis:** Lightweight traditional + MobileNetV3-Small deep features via HTTP download
3. **Feature Caching:** Both text and image features saved to `features/` as `.npy` files
4. **Feature Fusion:** Text and image features combined automatically at train/predict time
5. **Regression Modeling:** LightGBM with log-transform for robust price prediction

---

## 3. Model Architecture

### 3.1 Architecture Overview

```
Input Data (Text + Images)
         ↓
┌──────────────────────────────────────┐
│         src/feature_extraction.py    │
│  (shared by Light Mode + Cloud Mode) │
│                                      │
│  ┌─────────────────┬───────────────┐ │
│  │  Text Pipeline  │ Image Pipeline│ │
│  │                 │               │ │
│  │ • Word TF-IDF   │ • Brightness  │ │
│  │ • Char TF-IDF   │ • Contrast    │ │
│  │ • Brand Extract │ • Dominant RGB│ │
│  │ • Price Indicat │ • Texture Var │ │
│  │ • Qty/Size      │ • Aspect Ratio│ │
│  │ • Category      │ • Complexity  │ │
│  │ • KMeans Clust  │ • MobileNetV3 │ │
│  │ • Stat Features │   (576-dim)   │ │
│  └─────────────────┴───────────────┘ │
│         ↓ Saved to features/         │
└──────────────────────────────────────┘
         ↓
    Feature Fusion (auto at train time)
         ↓
    LightGBM Regressor
         ↓
    Price Prediction (log-space → expm1)
```

### 3.2 Model Components

**Text Processing Engine (`src/feature_extraction.py`):**
- Word-level TF-IDF (3000 features, unigrams + bigrams)
- Character-level TF-IDF (1000 features, 3-5 char ngrams)
- Brand extraction via regex patterns (Title case, "Brand:", "by X")
- Price indicator scoring (premium/luxury/budget keywords)
- Quantity extraction (pack size, count patterns)
- Size scoring (large/xl/jumbo vs small/mini/compact)
- Category classification (8 categories via keyword matching)
- KMeans clustering (20 clusters on SVD-reduced TF-IDF)
- Statistical text features (word lengths: mean, std, max, min)

**Image Processing Engine (`src/image_features_drive.py`):**
- Downloads images via parallel HTTP (ThreadPoolExecutor, 32 workers)
- Byte-level cache (capped at 10k entries) — no re-downloads
- 10 traditional features per image (single pass, no Canny/KMeans)
- MobileNetV3-Small deep features (576-dim, ~2.5MB weights)
- `torch.inference_mode()` for memory-efficient inference
- Batch size 256 for fast CPU/MPS/CUDA processing

**Regression Model:**
- LightGBM with L1 objective (MAE optimization)
- Log-transform (`log1p`) for price normalization
- Early stopping on validation set (Light mode)

---

## 4. Feature Engineering

### 4.1 Text Features (shared across all modes)

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Word TF-IDF | 3000 | Unigram + bigram, English stopwords removed |
| Char TF-IDF | 1000 | 3-5 char ngrams |
| Raw text stats | 8 | Length, word count, unique words, punctuation |
| Word length stats | 4 | Mean, std, max, min word length |
| Brand encoded | 1 | LabelEncoded brand from regex extraction |
| Price indicator | 1 | Keyword score (premium=+3, cheap=-2, etc.) |
| Quantity | 1 | Pack size from regex |
| Size score | 1 | Large/small keyword scoring |
| Category | 1 | 8-class keyword-based category |
| Cluster | 1 | KMeans cluster (20 clusters on SVD-50 TF-IDF) |
| **Total** | **4018** | All scaled via StandardScaler |

### 4.2 Image Features

| Feature | Count | Description |
|---------|-------|-------------|
| Brightness | 1 | Mean of RGB channel means |
| Contrast | 1 | Std of RGB channel means |
| Color variance | 1 | Variance across pixels (32×32 resize) |
| Dominant RGB | 3 | Mean color of resized image |
| Texture variance | 1 | Variance of grayscale image |
| Aspect ratio | 1 | Width / height |
| Center brightness | 1 | Mean of center 50% region |
| Image complexity | 1 | Std of full RGB array |
| MobileNetV3-Small | 576 | Deep embeddings, pretrained on ImageNet |
| **Total** | **586** | float32 |

---

## 5. Key Design Decisions

### 5.1 Shared Feature Extraction (`src/feature_extraction.py`)

Previously, `train_model_drive_light.py` had its own inline text extraction and `train_model_drive.py` used `src/text_features.py` — two different pipelines producing different features. Now both import from `src/feature_extraction.py`, ensuring:
- Identical features regardless of which script is used
- Single place to modify text/image extraction logic
- Shared `features/` cache folder

### 5.2 Feature Caching & Skip-if-Exists

```
features/
├── train_text.npy       # (50k × 4018) float64
├── test_text.npy        # (10k × 4018) float64
├── train_labels.npy     # (50k,) float64
├── test_ids.npy         # (10k,) object
├── text_encoders.pkl    # All fitted sklearn encoders
├── train_image.npy      # (50k × 586) float32  ← only if extracted
└── test_image.npy       # (10k × 586) float32  ← only if extracted
```

Re-running `extract_text` or `extract_image` when files exist prints:
```
✅ Text features already extracted — skipping.
✅ Image features already extracted — skipping.
```

### 5.3 MobileNetV3-Small vs DINOv2 vs ResNet50

| Model | Size | Dims | Batch (CPU) | Source |
|-------|------|------|-------------|--------|
| ResNet50 | ~100MB | 2048 | 32 | TensorFlow |
| DINOv2-small | ~85MB | 384 | 128 | torch.hub (download) |
| **MobileNetV3-Small** | **~2.5MB** | **576** | **256** | torchvision (bundled) |

MobileNetV3-Small was chosen for: smallest weights, no external download, largest batch size, ships with torchvision.

---

## 6. Execution Workflow

### 6.1 Commands

```bash
# Setup
python3 -m venv pricing_venv && source pricing_venv/bin/activate
pip install -r requirements.txt

# Feature extraction (one-time)
python train_model_drive_light.py extract_text    # ~5 min
python train_model_drive_light.py extract_image   # ~40-60 min (optional)

# Train & predict (repeat freely)
python train_model_drive_light.py train
python train_model_drive_light.py predict
python evaluate_results.py
```

Both `train_model_drive_light.py` and `train_model_drive.py` accept the same commands and share the same `features/` cache.

### 6.2 Workflow Diagram

```
First Run:
extract_text → [features/train_text.npy, features/test_text.npy, ...]
extract_image → [features/train_image.npy, features/test_image.npy]

Subsequent Runs:
extract_text  → "✅ already extracted — skipping"
extract_image → "✅ already extracted — skipping"

train  → loads features/ → trains LightGBM → saves models/
predict → loads features/ + models/ → saves dataset/test_out.csv
```

---

## 7. Performance

### 7.1 Validation Results

| Mode | Features | SMAPE | Training Time |
|------|----------|-------|---------------|
| Light — text only | 4018 | ~18-22% | ~10 min |
| Light — text + image | 4604 | ~15-18% | ~15 min |
| Cloud — text only | 4018 | ~18-22% | ~10 min |
| Cloud — text + image | 4604 | ~12-15% | ~15 min |
| Local — text + ResNet50 | 6066 | ~10-15% | ~30 min |

### 7.2 Feature Importance

- Text features contribute ~60-65% to model performance
- Brand encoding is the strongest single predictor
- MobileNetV3 deep features improve SMAPE by ~3-5% over text-only
- Semantic clustering (KMeans) reduces noise in text representations

---

## 8. Conclusion

**Key Innovations:**
- Unified `src/feature_extraction.py` shared across all pipelines
- Feature caching with skip-if-exists — no wasted computation on re-runs
- Lightweight MobileNetV3-Small replacing heavy DINOv2/ResNet50
- Separate `extract_text`, `extract_image`, `train`, `predict` commands for full control
- Automatic text+image fusion with text-only fallback

**Future Enhancements:**
- Integration of additional product metadata (reviews, ratings)
- Advanced ensemble methods combining multiple model architectures
- Enhanced image preprocessing for better feature extraction

---

## Appendix

### A. Code Structure

```
src/
├── feature_extraction.py     # Shared text + image extraction (NEW)
├── text_features.py          # Original text pipeline (reference)
├── image_features.py         # Local image feature extraction
├── image_features_drive.py   # Lightweight HTTP image extraction (MobileNetV3)
└── drive_utils_fast.py       # Drive API utilities

Main Scripts:
├── train_model.py             # Local mode (ResNet50, local images)
├── train_model_drive.py       # Cloud mode (shared feature_extraction)
├── train_model_drive_light.py # Light mode (shared feature_extraction)
├── data_setup.py              # Image download utility
└── evaluate_results.py        # SMAPE evaluation

Cache:
├── features/                  # Shared .npy feature cache
└── models/                    # Trained model .pkl files
```

### B. Bug Fixes Applied

1. `image_features_drive.py` — `BytesIO` resource leak fixed with `with` block + `.copy()`
2. `image_features_drive.py` — `torch.no_grad()` replaced with `torch.inference_mode()`
3. `image_features_drive.py` — Random seeds set for `torch`, `numpy`, `random`
4. `image_features_drive.py` — `_bytes_cache` capped at 10k entries
5. `train_model_drive.py` — Incomplete `train_model()` function fixed (was missing LightGBM training code)
