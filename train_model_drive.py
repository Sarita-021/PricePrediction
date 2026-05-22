import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

# Import feature extraction modules
from src.text_features import engineer_text_features
from src.image_features_drive import extract_comprehensive_image_features_drive

# --- Configuration ---
DATASET_FOLDER = 'dataset'
MODELS_FOLDER = 'models'
TRAIN_DATA_PATH = os.path.join(DATASET_FOLDER, 'train.csv')
TEST_DATA_PATH = os.path.join(DATASET_FOLDER, 'test.csv')
OUTPUT_PATH = os.path.join(DATASET_FOLDER, 'test_out.csv')

# Import configuration
try:
    from config_local import CREDENTIALS_PATH, FOLDER_ID
except ImportError:
    from config import CREDENTIALS_PATH, FOLDER_ID

FEATURES_DIR = 'features'
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def extract_text_features():
    """Extract and save text features for train and test sets"""
    print("📝 Extracting train text features...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    X_train_text, _, tfidf_vectorizer, feature_columns = engineer_text_features(
        train_df, fit_tfidf=True, analyze_importance=False
    )
    np.save(f'{FEATURES_DIR}/train_text.npy', X_train_text)
    np.save(f'{FEATURES_DIR}/train_labels.npy', train_df['price'].values)
    pickle.dump(tfidf_vectorizer, open(f'{FEATURES_DIR}/tfidf_vectorizer.pkl', 'wb'))
    pickle.dump(feature_columns, open(f'{FEATURES_DIR}/feature_columns.pkl', 'wb'))
    print(f"✓ Train text features saved: {X_train_text.shape}")

    print("📝 Extracting test text features...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test_text, _, _, _ = engineer_text_features(
        test_df, fit_tfidf=False, tfidf_vectorizer=tfidf_vectorizer, feature_columns=feature_columns
    )
    np.save(f'{FEATURES_DIR}/test_text.npy', X_test_text)
    np.save(f'{FEATURES_DIR}/test_ids.npy', test_df['sample_id'].values)
    print(f"✓ Test text features saved: {X_test_text.shape}")

def extract_image_features():
    """Extract and save image features for train and test sets"""
    print("🖼️ Extracting train image features...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    X_train_image, _ = extract_comprehensive_image_features_drive(
        train_df, use_deep_features=True,
        credentials_path=CREDENTIALS_PATH, folder_id=FOLDER_ID
    )
    np.save(f'{FEATURES_DIR}/train_image.npy', X_train_image.astype(np.float32))
    print(f"✓ Train image features saved: {X_train_image.shape}")

    print("🖼️ Extracting test image features...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test_image, _ = extract_comprehensive_image_features_drive(
        test_df, use_deep_features=True,
        credentials_path=CREDENTIALS_PATH, folder_id=FOLDER_ID
    )
    np.save(f'{FEATURES_DIR}/test_image.npy', X_test_image.astype(np.float32))
    print(f"✓ Test image features saved: {X_test_image.shape}")

def _load_features(split='train'):
    """Load and combine saved text + image features if available"""
    text = np.load(f'{FEATURES_DIR}/{split}_text.npy')
    img_path = f'{FEATURES_DIR}/{split}_image.npy'
    if os.path.exists(img_path):
        img = np.load(img_path)
        print(f"Combining text {text.shape} + image {img.shape} features")
        return np.hstack([text, img])
    print(f"No image features found, using text only: {text.shape}")
    return text

def train_model():
    """Train model from saved features"""
    if not os.path.exists(f'{FEATURES_DIR}/train_text.npy'):
        print("❌ Text features not found. Run: python train_model_drive.py extract_text")
        return False

    print("🚀 Loading saved features...")
    X_train = _load_features('train')
    Y_train_log = np.log1p(np.load(f'{FEATURES_DIR}/train_labels.npy'))
    print(f"✓ Training features shape: {X_train.shape}")

def predict_test_data():
    """Generate predictions from saved features"""
    if not os.path.exists(f'{FEATURES_DIR}/test_text.npy'):
        print("❌ Text features not found. Run: python train_model_drive.py extract_text")
        return False

    print("\n🔮 Loading saved features...")
    X_test = _load_features('test')
    sample_ids = np.load(f'{FEATURES_DIR}/test_ids.npy')

    try:
        model = pickle.load(open(os.path.join(MODELS_FOLDER, 'trained_model.pkl'), 'rb'))
        metadata = pickle.load(open(os.path.join(MODELS_FOLDER, 'model_metadata.pkl'), 'rb'))
        print(f"✓ Loaded model with {metadata['total_features']} features")
    except FileNotFoundError:
        print("❌ ERROR: Trained model not found. Run training first.")
        return False

    # Align feature dimensions
    expected = metadata['total_features']
    if X_test.shape[1] < expected:
        X_test = np.hstack([X_test, np.zeros((X_test.shape[0], expected - X_test.shape[1]))])
    elif X_test.shape[1] > expected:
        X_test = X_test[:, :expected]

    print("\n🎯 Generating Predictions...")
    Y_pred = np.clip(np.expm1(model.predict(X_test)), a_min=0, a_max=None)

    pd.DataFrame({'sample_id': sample_ids, 'price': Y_pred}).to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Predictions saved to {OUTPUT_PATH}")
    print(f"✓ Price range: ${Y_pred.min():.2f} - ${Y_pred.max():.2f}")
    return True

def train_and_predict_pipeline():
    """Complete pipeline using Google Drive"""
    print("🚀 Starting Complete ML Pipeline with Google Drive...")
    
    if not train_model():
        print("❌ Training failed!")
        return
    
    if not predict_test_data():
        print("❌ Prediction failed!")
        return
    
    print("\n🎉 Complete pipeline finished successfully!")

if __name__ == "__main__":
    import sys

    commands = {
        'extract_text':  extract_text_features,
        'extract_image': extract_image_features,
        'train':         train_model,
        'predict':       predict_test_data,
        'full':          lambda: (extract_text_features(), train_model(), predict_test_data()),
    }

    if len(sys.argv) > 1 and sys.argv[1] in commands:
        commands[sys.argv[1]]()
    else:
        print("Usage: python train_model_drive.py [command]")
        print("Commands:")
        print("  extract_text   — extract & save text features for train+test")
        print("  extract_image  — extract & save image features for train+test")
        print("  train          — train model from saved features")
        print("  predict        — generate predictions from saved features")
        print("  full           — extract_text + train + predict")