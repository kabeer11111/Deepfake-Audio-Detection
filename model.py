import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import numpy as np
import os

MODEL_PATH = "deepfake_detector_final.h5"
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_audio(filepath):
    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]
    return np.expand_dims(mel_spectrogram, axis=[0, -1]), audio

def plot_frequency_graph(audio, filename):
    plt.figure(figsize=(10, 4))
    plt.specgram(audio, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512)
    plt.title('Frequency Spectrum')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.savefig(os.path.join('static', filename))
    plt.close()

def predict_audio(filepath):
    model = load_model()
    audio_data, audio = preprocess_audio(filepath)
    prediction = model.predict(audio_data)
    prediction_class = np.argmax(prediction, axis=1)[0]
    plot_frequency_graph(audio, 'frequency_graph.png')
    return "Bonafide" if prediction_class == 1 else "Spoof"
