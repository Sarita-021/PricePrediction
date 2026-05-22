import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import re

FEATURES_DIR = 'features'

TEXT_FILES   = ['train_text.npy', 'test_text.npy', 'train_labels.npy', 'test_ids.npy', 'text_encoders.pkl']
IMAGE_FILES  = ['train_image.npy', 'test_image.npy']

def _text_features_exist():
    return all(os.path.exists(os.path.join(FEATURES_DIR, f)) for f in TEXT_FILES)

def _image_features_exist():
    return all(os.path.exists(os.path.join(FEATURES_DIR, f)) for f in IMAGE_FILES)


# ── Text feature extraction (shared by both pipelines) ───────────────────────

def _extract_advanced_features(df, encoders=None, fit=True):
    if fit:
        encoders = {}

    if fit:
        encoders['tfidf_word'] = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
        word_tfidf = encoders['tfidf_word'].fit_transform(df['catalog_content'].fillna(''))
    else:
        word_tfidf = encoders['tfidf_word'].transform(df['catalog_content'].fillna(''))

    if fit:
        encoders['tfidf_char'] = TfidfVectorizer(max_features=1000, analyzer='char', ngram_range=(3, 5))
        char_tfidf = encoders['tfidf_char'].fit_transform(df['catalog_content'].fillna(''))
    else:
        char_tfidf = encoders['tfidf_char'].transform(df['catalog_content'].fillna(''))

    text_features, brands, price_indicators, quantities, measurements, categories = [], [], [], [], [], []

    brand_patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'Brand:\s*([A-Za-z\s]+)',
        r'by\s+([A-Z][a-z]+)',
    ]
    price_words = {
        'premium': 3, 'luxury': 4, 'deluxe': 3, 'professional': 2,
        'pro': 2, 'advanced': 2, 'basic': -1, 'standard': 0,
        'cheap': -2, 'budget': -1, 'economy': -1, 'value': -1
    }
    detailed_categories = {
        'electronics': ['electronic', 'digital', 'tech', 'device', 'gadget', 'computer', 'phone'],
        'clothing': ['shirt', 'dress', 'clothing', 'apparel', 'fashion', 'wear', 'fabric'],
        'home_kitchen': ['home', 'kitchen', 'furniture', 'decor', 'appliance'],
        'beauty_health': ['beauty', 'cosmetic', 'skincare', 'health', 'wellness'],
        'sports_outdoor': ['sport', 'fitness', 'exercise', 'outdoor', 'athletic'],
        'books_media': ['book', 'dvd', 'cd', 'media', 'magazine'],
        'toys_games': ['toy', 'game', 'play', 'puzzle', 'doll'],
        'automotive': ['car', 'auto', 'vehicle', 'motor', 'tire']
    }

    for text in df['catalog_content'].fillna(''):
        tl = text.lower()
        words = text.split()
        word_lengths = [len(w) for w in words] if words else [0]

        text_features.append([
            len(text), len(words), len(set(words)),
            text.count('.'), text.count(','),
            len(re.findall(r'\d+', text)), len(re.findall(r'[A-Z]', text)),
            text.count('!') + text.count('?'),
            np.mean(word_lengths), np.std(word_lengths) if len(word_lengths) > 1 else 0,
            max(word_lengths), min(word_lengths),
        ])

        found_brand = 'Unknown'
        for pattern in brand_patterns:
            m = re.search(pattern, text)
            if m:
                found_brand = m.group(1).strip()
                break
        brands.append(found_brand)

        price_indicators.append(sum(price_words.get(w, 0) for w in tl.split()))

        qty = 1
        for pattern in [r'(\d+)\s*(?:pack|pcs|pieces|count|ct|box)', r'pack\s*of\s*(\d+)', r'(\d+)\s*in\s*1']:
            m = re.search(pattern, tl)
            if m:
                qty = int(m.group(1))
                break
        quantities.append(qty)

        size_score = sum(2 for w in ['large', 'xl', 'big', 'jumbo', 'giant', 'mega'] if w in tl)
        size_score -= sum(1 for w in ['small', 'mini', 'tiny', 'compact'] if w in tl)
        measurements.append(size_score)

        found_cat, max_matches = 'other', 0
        for cat, keywords in detailed_categories.items():
            matches = sum(1 for kw in keywords if kw in tl)
            if matches > max_matches:
                max_matches, found_cat = matches, cat
        categories.append(found_cat)

    if fit:
        encoders['brand'] = LabelEncoder()
        brand_encoded = encoders['brand'].fit_transform(brands)
        encoders['category'] = LabelEncoder()
        cat_encoded = encoders['category'].fit_transform(categories)
        encoders['svd'] = TruncatedSVD(n_components=50, random_state=42)
        reduced = encoders['svd'].fit_transform(word_tfidf)
        encoders['kmeans'] = KMeans(n_clusters=20, random_state=42, n_init=10)
        clusters = encoders['kmeans'].fit_predict(reduced)
    else:
        brand_encoded = np.array([
            encoders['brand'].transform([b])[0] if b in encoders['brand'].classes_ else 0
            for b in brands
        ])
        cat_encoded = np.array([
            encoders['category'].transform([c])[0] if c in encoders['category'].classes_ else 0
            for c in categories
        ])
        reduced  = encoders['svd'].transform(word_tfidf)
        clusters = encoders['kmeans'].predict(reduced)

    numerical = np.hstack([
        np.array(text_features),
        np.column_stack([brand_encoded, price_indicators, quantities, measurements, cat_encoded, clusters])
    ])

    if fit:
        encoders['scaler'] = StandardScaler()
        numerical = encoders['scaler'].fit_transform(numerical)
    else:
        numerical = encoders['scaler'].transform(numerical)

    return np.hstack([word_tfidf.toarray(), char_tfidf.toarray(), numerical]), encoders


# ── Shared extract_text_features ─────────────────────────────────────────────

def extract_text_features():
    """Extract and cache text features (shared by both pipelines)."""
    if _text_features_exist():
        print("✅ Text features already extracted — skipping.")
        return

    os.makedirs(FEATURES_DIR, exist_ok=True)

    print("📝 Extracting train text features...")
    train_df = pd.read_csv('dataset/train.csv')
    Q1, Q3 = train_df['price'].quantile(0.05), train_df['price'].quantile(0.95)
    train_df = train_df[(train_df['price'] >= Q1) & (train_df['price'] <= Q3)]
    X_train, encoders = _extract_advanced_features(train_df, fit=True)
    np.save(f'{FEATURES_DIR}/train_text.npy', X_train)
    np.save(f'{FEATURES_DIR}/train_labels.npy', train_df['price'].values)
    pickle.dump(encoders, open(f'{FEATURES_DIR}/text_encoders.pkl', 'wb'))
    print(f"✓ Train text features saved: {X_train.shape}")

    print("📝 Extracting test text features...")
    test_df = pd.read_csv('dataset/test.csv')
    X_test, _ = _extract_advanced_features(test_df, encoders, fit=False)
    np.save(f'{FEATURES_DIR}/test_text.npy', X_test)
    np.save(f'{FEATURES_DIR}/test_ids.npy', test_df['sample_id'].values)
    print(f"✓ Test text features saved: {X_test.shape}")


# ── Shared extract_image_features ────────────────────────────────────────────

def extract_image_features():
    """Extract and cache image features (shared by both pipelines)."""
    if _image_features_exist():
        print("✅ Image features already extracted — skipping.")
        return

    from src.image_features_drive import extract_comprehensive_image_features_drive
    os.makedirs(FEATURES_DIR, exist_ok=True)

    print("🖼️  Extracting train image features...")
    train_df = pd.read_csv('dataset/train.csv')
    Q1, Q3 = train_df['price'].quantile(0.05), train_df['price'].quantile(0.95)
    train_df = train_df[(train_df['price'] >= Q1) & (train_df['price'] <= Q3)]
    X_train_img, _ = extract_comprehensive_image_features_drive(train_df)
    np.save(f'{FEATURES_DIR}/train_image.npy', X_train_img.astype(np.float32))
    print(f"✓ Train image features saved: {X_train_img.shape}")

    print("🖼️  Extracting test image features...")
    test_df = pd.read_csv('dataset/test.csv')
    X_test_img, _ = extract_comprehensive_image_features_drive(test_df)
    np.save(f'{FEATURES_DIR}/test_image.npy', X_test_img.astype(np.float32))
    print(f"✓ Test image features saved: {X_test_img.shape}")


# ── Shared feature loader ─────────────────────────────────────────────────────

def load_features(split='train'):
    """Load and combine saved text + image features if available."""
    text = np.load(f'{FEATURES_DIR}/{split}_text.npy')
    img_path = f'{FEATURES_DIR}/{split}_image.npy'
    if os.path.exists(img_path):
        img = np.load(img_path)
        print(f"Combining text {text.shape} + image {img.shape} features")
        return np.hstack([text, img])
    print(f"No image features found, using text only: {text.shape}")
    return text
