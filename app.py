import os
import numpy as np 
import librosa 
import soundfile 
import joblib
from flask import Flask, request, jsonify, render_template 
from werkzeug.utils import secure_filename


app = Flask(__name__) # Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'} 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Load the trained model (path updated to reflect the structure)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'mlp_model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'scaler.joblib')
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
  raise FileNotFoundError(f"Model or scaler file not found. Please ensure both files are saved correctly.")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# Define the emotions (same as in training)
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


# Function to check allowed file extensions 
def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to extract features from audio (same as in training script)

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    try:
        X, sample_rate = librosa.load(file_name, sr=22050, mono=True)
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

    stft = np.abs(librosa.stft(X))
    features = []

    # 1. Chroma must come first to match training!
    if chroma:
        chroma_feat = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        features.extend(np.mean(chroma_feat.T, axis=0))

    # 2. MFCC second
    if mfcc:
        mfcc_feat = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        features.extend(np.mean(mfcc_feat.T, axis=0))

    # 3. Mel third
    if mel:
        mel_feat = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        features.extend(np.mean(mel_feat.T, axis=0))

    return np.array(features)


# Route to serve the frontend 
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle audio upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
      return jsonify({'error': 'No file part in the request'}), 400


    file = request.files['file']


    if file.filename == '':
      return jsonify({'error': 'No file selected'}), 400


    if not allowed_file(file.filename):
      return jsonify({'error': 'Unsupported file format. Please upload a WAV, MP3, OGG, or FLAC file.'}),400


    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
    file.save(filepath)

    try:
      features = extract_feature(filepath, mfcc=True, chroma=True, mel=True)
      if features is None:
        return jsonify({'error': 'Failed to process the audio file'}), 500

      features = features.reshape(1, -1)
      features = scaler.transform(features)   # <-- critical for MLP
      prediction = model.predict(features)

      return jsonify({'emotion': str(prediction[0])})
 

    except Exception as e:
      return jsonify({'error': str(e)}), 500 

    finally:
      if os.path.exists(filepath): 
        os.remove(filepath)

if __name__ == '__main__': 
  app.run(debug=True)



