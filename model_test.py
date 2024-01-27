import librosa
import pandas as pd
import numpy as np
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
import sounddevice as sd
import noisereduce


# Function to extract features
def extract_features(x, mmfc=False):
    mmfc_dict = {}

    if (not mmfc):
        mmfc = librosa.feature.mfcc(y=x, sr=44100, n_mfcc=20)
        for i in range(mmfc.shape[0]):
            mmfc_stats = extract_features(mmfc[i], mmfc=True)
            for key in mmfc_stats.keys():
                mmfc_dict[f"mmfc{i}_{key}"] = mmfc_stats[key]
    fmax = np.max(x)
    fmin = np.min(x)
    return {
        **mmfc_dict,
        "mean": np.mean(x),
        "std": np.std(x),
        "max": fmax,
        "min": fmin,
        "kurtosis": scipy.stats.kurtosis(x),
        "peak_to_peak": fmax-fmin,
        "zero": sum(librosa.zero_crossings(x)) / len(x),
        "rms": np.sqrt(np.mean(np.square(x))),
        "crest": np.max(np.abs(x))/np.sqrt(np.mean(np.square(x))),
    }


# Load dataset
df = pd.read_csv("./data/dataset.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# Train model
model = RandomForestClassifier()
model.fit(X, y)

samplerate = 44100
duration = 2


# Testing loop
while True:

    input("Press ENTER to record...")
    print("\nRecording...")
    duration_ = int(samplerate * duration)
    rec = sd.rec(duration_, samplerate=samplerate,
                 channels=1, blocking=True, dtype="int16")
    print("Recorded\n")

    reduced_ = noisereduce.reduce_noise(rec.flatten(), samplerate)
    trimmed_ = librosa.effects.trim(reduced_, top_db=20)[0]
    normalized_ = normalize([trimmed_])[0]

    features = extract_features(normalized_)
    features = pd.DataFrame([features])

    predicted = model.predict(features)[0]
    print("Prediction: ", predicted, end="\n\n\n")
