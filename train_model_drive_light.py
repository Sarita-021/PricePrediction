import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from src.feature_extraction import extract_text_features, extract_image_features, load_features, FEATURES_DIR
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def train_model():
    """Train model from saved features"""
    if not os.path.exists(f'{FEATURES_DIR}/train_text.npy'):
        print("❌ Text features not found. Run: python train_model_drive_light.py extract_text")
        return

    print("Loading saved features...")
    X = load_features('train')
    y_log = np.log1p(np.load(f'{FEATURES_DIR}/train_labels.npy'))
    print(f"Feature shape: {X.shape}")
    
    # Train single model
    print("Training LightGBM...")
    X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=127,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=1,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
    )
    
    # Evaluate single model
    pred = np.expm1(lgb_model.predict(X_val))
    y_val_original = np.expm1(y_val)
    score = smape(y_val_original, pred)
    print(f"Validation SMAPE: {score:.2f}%")

    import matplotlib.pyplot as plt
    lgb.plot_importance(lgb_model, importance_type='gain', max_num_features=30, figsize=(10, 6), title='Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig('models/feature_importance_gain.png')

    lgb.plot_importance(lgb_model, importance_type='split', max_num_features=30, figsize=(10, 6), title='Feature Importance (Split)')
    plt.tight_layout()
    plt.savefig('models/feature_importance_split.png')
    plt.show()

    os.makedirs('models', exist_ok=True)
    pickle.dump(lgb_model, open('models/advanced_model.pkl', 'wb'))
    print("✅ Model saved!")

def predict():
    """Generate predictions from saved features"""
    if not os.path.exists(f'{FEATURES_DIR}/test_text.npy'):
        print("❌ Text features not found. Run: python train_model_drive_light.py extract_text")
        return

    print("Loading saved features...")
    X_test = load_features('test')
    sample_ids = np.load(f'{FEATURES_DIR}/test_ids.npy')

    model = pickle.load(open('models/advanced_model.pkl', 'rb'))
    expected = model.n_features_in_
    if X_test.shape[1] < expected:
        X_test = np.hstack([X_test, np.zeros((X_test.shape[0], expected - X_test.shape[1]))])
    elif X_test.shape[1] > expected:
        X_test = X_test[:, :expected]
    pred = np.expm1(model.predict(X_test))
    pred = np.clip(pred, 0.01, None)

    pd.DataFrame({'sample_id': sample_ids, 'price': pred}).to_csv('dataset/test_out.csv', index=False)
    print(f"Predictions saved! Price range: ${pred.min():.2f} - ${pred.max():.2f}")

if __name__ == "__main__":
    import sys

    commands = {
        'extract_text':  extract_text_features,
        'extract_image': extract_image_features,
        'train':         train_model,
        'predict':       predict,
        'full':          lambda: (extract_text_features(), train_model(), predict()),
    }

    if len(sys.argv) > 1 and sys.argv[1] in commands:
        commands[sys.argv[1]]()
    else:
        print("Usage: python train_model_drive_light.py [command]")
        print("Commands:")
        print("  extract_text   — extract & save text features for train+test")
        print("  extract_image  — extract & save image features for train+test")
        print("  train          — train model from saved features")
        print("  predict        — generate predictions from saved features")
        print("  full           — extract_text + train + predict")