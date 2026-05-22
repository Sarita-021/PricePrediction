# ML Challenge 2025 - Smart Product Pricing

## Smart Product Pricing Challenge

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. Your challenge is to develop an ML solution that analyzes product details and predict the price of the product. The relationship between product attributes and pricing is complex - with factors like brand, specifications, product quantity directly influence pricing. Your task is to build a model that can analyze these product details holistically and suggest an optimal price.

### Data Description:

The dataset consists of the following columns:

1. **sample_id:** A unique identifier for the input sample
2. **catalog_content:** Text field containing title, product description and an Item Pack Quantity(IPQ) concatenated.
3. **image_link:** Public URL where the product image is available for download.
   Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
   To download images use `download_images` function from `src/utils.py`. See sample code in `src/test.ipynb`.
4. **price:** Price of the product (Target variable - only available in training data)

### Dataset Details:

- **Training Dataset:** 50k products with complete product details and prices
- **Test Set:** 10k products for final evaluation

### Output Format:

The output file should be a CSV with 2 columns:

1. **sample_id:** The unique identifier of the data sample. Note the ID should match the test record sample_id.
2. **price:** A float value representing the predicted price of the product.

Note: Make sure to output a prediction for all sample IDs. If you have less/more number of output samples in the output file as compared to test.csv, your output won't be evaluated.

### File Descriptions:

*Source files*

1. **src/feature_extraction.py:** Shared text and image feature extraction used by both Light and Cloud pipelines. Handles caching to disk and skip-if-already-extracted logic.
2. **src/text_features.py:** Original advanced text feature engineering (used as reference).
3. **src/image_features.py:** Local image feature extraction including color, texture, composition, and ResNet50/EfficientNet deep features.
4. **src/image_features_drive.py:** Lightweight image feature extraction via HTTP — MobileNetV3-Small (576-dim) deep features + 10 traditional features.
5. **src/drive_utils_fast.py:** Google Drive integration utilities for fast cloud-based processing.

*Dataset files*

1. **dataset/train.csv:** Training file with labels (`price`).
2. **dataset/test.csv:** Test file without output labels (`price`). Generate predictions using your model/solution on this file's data.
3. **dataset/test_out_correct.csv:** Test file with correct prices for SMAPE evaluation during development.

---

### Evaluation Criteria:

Submissions are evaluated using **Symmetric Mean Absolute Percentage Error (SMAPE)**: A statistical measure that expresses the relative difference between predicted and actual values as a percentage, while treating positive and negative errors equally.

**Formula:**
```
SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)
```

**Example:** If actual price = $100 and predicted price = $120
SMAPE = |100-120| / ((|100| + |120|)/2) * 100% = 18.18%

**Note:** SMAPE is bounded between 0% and 200%. Lower values indicate better performance.

---

## 🚀 ML Pipeline Execution Guide

### **Prerequisites / Setup**
```bash
# Create virtual environment
python3 -m venv pricing_venv

# Activate virtual environment
source pricing_venv/bin/activate  # On macOS/Linux
# OR
pricing_venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

### **Execution Modes**

#### **⚡ LIGHT MODE (`train_model_drive_light.py`)**
**Best for:** Limited memory systems, quick iteration, no local images needed

```bash
# Step 1 — Extract text features (one-time, ~5 min)
python train_model_drive_light.py extract_text

# Step 2 — Extract image features (one-time, ~40-60 min, optional)
python train_model_drive_light.py extract_image

# Step 3 — Train model (repeat freely without re-extracting)
python train_model_drive_light.py train

# Step 4 — Generate predictions
python train_model_drive_light.py predict

# Step 5 — Evaluate SMAPE score (optional)
python evaluate_results.py
```

#### **☁️ CLOUD MODE (`train_model_drive.py`)**
**Best for:** Full dataset, higher accuracy, cloud environments

```bash
# Step 1 — Extract text features (one-time, ~5 min)
python train_model_drive.py extract_text

# Step 2 — Extract image features (one-time, ~40-60 min, optional)
python train_model_drive.py extract_image

# Step 3 — Train model
python train_model_drive.py train

# Step 4 — Generate predictions
python train_model_drive.py predict

# Step 5 — Evaluate SMAPE score (optional)
python evaluate_results.py
```

#### **🎯 LOCAL MODE (`train_model.py`)**
**Best for:** Production use, highest accuracy, images downloaded locally

```bash
# Step 1 — Download images
python data_setup.py

# Step 2 — Train model
python train_model.py train

# Step 3 — Generate predictions
python train_model.py predict

# Step 4 — Evaluate SMAPE score (optional)
python evaluate_results.py
```

> **Note:** `extract_text` and `extract_image` only need to run **once**. Re-running them prints `✅ already extracted — skipping`. Both Light and Cloud modes share the same `features/` cache folder — if you've extracted features using one script, the other will reuse them automatically.

---

### **Available Commands**

| Command | Description | Run Once? |
|---------|-------------|-----------|
| `extract_text` | Extract & cache text features for train+test | ✅ Yes |
| `extract_image` | Extract & cache image features for train+test | ✅ Yes |
| `train` | Train LightGBM from cached features | 🔁 Repeat freely |
| `predict` | Generate predictions from cached features | 🔁 Repeat freely |
| `full` | extract_text + train + predict in one go | — |

---

### **Key Features & Innovations**

- **Shared Feature Extraction:** `src/feature_extraction.py` is used by both Light and Cloud pipelines — same text features, no duplication
- **Feature Caching:** Text and image features saved to `features/` as `.npy` files — extract once, reuse forever
- **Skip-if-Exists:** Re-running `extract_text` or `extract_image` detects existing files and skips automatically
- **Automatic Fallback:** If image features aren't extracted, `train` and `predict` fall back to text-only mode
- **Lightweight Image Model:** `image_features_drive.py` uses MobileNetV3-Small (~2.5MB) instead of DINOv2/ResNet50 — 9x smaller, faster on CPU
- **Advanced Text Processing:** TF-IDF (word + char level), brand extraction, semantic clustering, price indicators
- **Memory Optimization:** Chunked processing, capped image byte cache (10k entries), `torch.inference_mode()`
- **Robust Error Handling:** Graceful fallbacks for missing images/data

---

### **File Structure After Execution**

```
PricePrediction/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── test_out.csv              # Generated predictions
│   └── test_out_correct.csv
├── features/                     # Cached features (shared by Light + Cloud)
│   ├── train_text.npy
│   ├── test_text.npy
│   ├── train_labels.npy
│   ├── test_ids.npy
│   ├── text_encoders.pkl
│   ├── train_image.npy           # Only if extract_image was run
│   └── test_image.npy            # Only if extract_image was run
├── models/
│   ├── trained_model.pkl         # LightGBM model (Cloud mode)
│   ├── advanced_model.pkl        # LightGBM model (Light mode)
│   └── model_metadata.pkl
├── images/                       # Downloaded images (LOCAL mode only)
│   ├── train/
│   └── test/
└── src/
    ├── feature_extraction.py     # Shared text + image extraction
    ├── text_features.py
    ├── image_features.py
    ├── image_features_drive.py
    └── drive_utils_fast.py
```

---

### **Mode Selection Guide**

| Mode | Time | Memory | Accuracy | Dataset | Use Case |
|------|------|--------|----------|---------|----------|
| **Local** | 2-4 hours | High (16GB+) | Highest | Full (50k) | Production/Best Results |
| **Light** | 30-60 min | Low (4-8GB) | Good | Full (50k) | Quick Testing/Limited RAM |
| **Cloud** | 1-2 hours | Medium (8-16GB) | High | Full (50k) | Cloud Computing |

### **Feature Comparison**

| Feature | Local | Light | Cloud |
|---------|-------|-------|-------|
| Text Processing | ✅ Full TF-IDF | ✅ Full TF-IDF (shared) | ✅ Full TF-IDF (shared) |
| Image Source | ✅ Local Files | ✅ HTTP Download | ✅ HTTP Download |
| Deep Learning | ✅ ResNet50 (2048-dim) | ✅ MobileNetV3 (576-dim) | ✅ MobileNetV3 (576-dim) |
| Semantic Clustering | ✅ Yes | ✅ Yes | ✅ Yes |
| Brand Extraction | ✅ Yes | ✅ Yes | ✅ Yes |
| Feature Caching | ❌ No | ✅ Yes | ✅ Yes |
| Skip-if-Exists | ❌ No | ✅ Yes | ✅ Yes |
| Memory Efficiency | ❌ High Usage | ✅ Optimized | ✅ Optimized |

---

### **Troubleshooting**

#### **Memory Issues**
- **"zsh: killed" error:** System OOM killer terminated process
- **Solution:** Use Light Mode, reduce `batch_size` in `image_features_drive.py`
- **Check RAM:** Ensure 4GB+ free for Light mode

#### **Feature Extraction Issues**
- **Re-running extract_text/extract_image:** Safe — will print `✅ already extracted — skipping`
- **Want to re-extract:** Delete files in `features/` folder and re-run

#### **Model Loading Errors**
- **"Trained model not found":** Run `train` before `predict`
- **Check:** Verify `models/` folder contains `.pkl` files

#### **Image Download Failures**
- Script continues with zero-filled features for failed downloads
- Check internet connection if many images fail

#### **Dependency Issues**
```bash
pip install -r requirements.txt
pip install torch torchvision        # For MobileNetV3 image features
pip install opencv-python            # For traditional image features
```

#### **Common Error Messages**
```bash
# Memory exhaustion
zsh: killed → Reduce batch_size or use Light Mode

# Missing features
Text features not found → Run extract_text first

# Missing model
FileNotFoundError: trained_model.pkl → Run train first

# PyTorch GPU (MPS/CUDA)
Could not load dynamic library → CPU fallback (normal)
```

---

## **Quick Start Recommendations**

1. **First Time / Limited Memory:** Use Light Mode with `extract_text` only (skip `extract_image`)
2. **Best Accuracy:** Run both `extract_text` and `extract_image`, then `train`
3. **Iterating on model:** Run `extract_text` + `extract_image` once, then `train` as many times as needed
4. **Switching between Light and Cloud:** Features are shared — no need to re-extract

## **Performance Expectations**

| Mode | Text Only | Text + Images |
|------|-----------|---------------|
| **Light** | ~18-22% SMAPE | ~15-18% SMAPE |
| **Cloud** | ~18-22% SMAPE | ~12-15% SMAPE |
| **Local** | — | ~10-15% SMAPE |

- **Processing Time:** extract_text (~5 min) + extract_image (~40-60 min) + train (~10-20 min)
- **Memory Usage:** Light (4-8GB) vs Cloud (8-16GB) vs Local (16GB+)
