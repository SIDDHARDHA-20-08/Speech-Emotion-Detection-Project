import argparse
import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib


def extract_feature(file_path, mfcc=True, chroma=True, mel=True):
    """
    Extract audio features from a wav file:
    - MFCC
    - Chroma
    - MEL Spectrogram
    """
    try:
        X, sample_rate = librosa.load(file_path, sr=22050, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    stft = np.abs(librosa.stft(X))
    features = []

    if chroma:
        chroma_feat = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        features.extend(np.mean(chroma_feat.T, axis=0))

    if mfcc:
        mfcc_feat = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        features.extend(np.mean(mfcc_feat.T, axis=0))

    if mel:
        mel_feat = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        features.extend(np.mean(mel_feat.T, axis=0))

    return np.array(features)


def load_data(data_dir):
    """
    Traverse data_dir and extract features and labels from WAV files.
    Assumes filenames follow the RAVDESS convention,
    where the 3rd segment is the emotion code.
    """
    emotion = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    X, y = [], []

    for root, _, files in os.walk(data_dir):
        emotion = os.path.basename(root).lower()
        if emotion not in ['angry','sad','happy','fear','neutral','disgust']:
            continue
        for fname in files:
            if not fname.lower().endswith('.wav'):
                continue
            file_path = os.path.join(root, fname)
            feat = extract_feature(file_path)
            if feat is not None:
                X.append(feat)
                y.append(emotion)
    return np.array(X), np.array(y)


def main(args):
    # Load and split data
    print(f"Loading data from: {args.data_dir}")
    X, y = load_data(args.data_dir)

    if len(y) == 0:
        print("No valid audio files found. Check data_dir and file naming.")
        return
    import collections
    print(collections.Counter(y))

    print(f"Total samples: {len(y)}")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    # Apply scaling for MLP
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # Train classifier
    print("Training MLP classifier...")
    model = MLPClassifier(hidden_layer_sizes=(512,256,128), max_iter=1000, random_state=42)

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Dump the trained model using joblib
    print(f"Saving trained model to: {args.model_path}")
    joblib.dump(model, args.model_path)
    scaler_path = os.path.join(os.path.dirname(args.model_path), "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print("Model dumped successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train emotion detection and dump model")
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help="Root folder containing Actor_## subfolders"
    )
    parser.add_argument(
        '--test_size', type=float, default=0.2,
        help="Fraction of data for evaluation"
    )
    parser.add_argument(
        '--random_state', type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        '--model_path', type=str, default='emotion_model.joblib',
        help="File path to dump the trained model"
    )

    args = parser.parse_args()
    main(args)
