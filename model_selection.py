# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 20:59:58 2024

@author: Ersin Namal
"""
import noisereduce
import sounddevice as sd
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import pandas as pd
import scipy
from pydub import AudioSegment
from pydub.playback import play


# Read audio file
sr, x = wavfile.read("./data/train_data2.wav")


# Plot raw signal
plt.title("Raw signal")
time_axis = 1 * np.arange(0, len(x), 1) / float(sr)
plt.plot(time_axis, x, color="#343a40")
plt.xlabel("seconds")
plt.show()


# Noise reduction
reduced = noisereduce.reduce_noise(y=x, sr=sr, n_jobs=-1, n_fft=256)
plt.plot(time_axis, reduced, color="#343a40")

plt.title("Signal after noise reduction")
plt.xlabel("seconds")
plt.show()


# Remove signals after the 40th second
x = reduced[:40 * sr]
time_axis = 1 * np.arange(0, len(x), 1) / float(sr)

plt.plot(time_axis, x, color="#343a40")
plt.title("Trimmed signal")
plt.xlabel("seconds")
plt.show()


# Find intervals with possible meaningful data
intervals = librosa.effects.split(x)


# Labels
labels = ["COMPUTER", "ENGINEERING", "ERSIN", "NAMAL", "COUGH", "CLAP", "SNAP"]


# Function to print labels when labeling data
def print_options():
    for i in range(len(labels)):
        print(f"({i}) {labels[i]}", end="\n")
    print("\n(Q) Skip")


labeled_data = []

# Label the data
for interval in intervals:

    start, end = interval
    sound = AudioSegment(x[start:end], frame_rate=sr,
                         sample_width=2, channels=1)
    play(sound)

    plt.plot(x[start:end])
    plt.title("Frame")
    plt.show()

    while True:
        print_options()
        choice = input("\n> ").upper()

        if (choice == "Q"):
            break
        try:
            choice = int(choice)
            labeled_data.append((labels[choice], [start, end]))
            break
        except:
            continue


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


X = []
y = []

# Extract features
for i in range(len(labeled_data)):

    start, end = labeled_data[i][1]
    sound = x[start:end]

    normalized = normalize([sound])[0]

    X.append(extract_features(normalized))
    y.append(labeled_data[i][0])


df = pd.DataFrame(X)
df["label"] = y


df.to_csv("./data/dataset.csv", index=False)

models = {"RandomForest": RandomForestClassifier(),
          "KNeighbors": KNeighborsClassifier(),
          "AdaBoost": AdaBoostClassifier()}

for key in models.keys():
    models[key].fit(df.iloc[:, :-1], df.iloc[:, -1])

samplerate = 44100
duration = 2

while True:

    input("\n\n\nTrue class     : ")
    print()

    duration_ = int(samplerate * duration)
    rec = sd.rec(duration_, samplerate=samplerate,
                 channels=1, blocking=True, dtype="int16")

    reduced_ = noisereduce.reduce_noise(rec.flatten(), samplerate)
    trimmed_ = librosa.effects.trim(reduced_, top_db=20)[0]
    normalized_ = normalize([trimmed_])[0]

    features = extract_features(normalized_)
    features = pd.DataFrame([features])

    for key in models.keys():
        predicted = models[key].predict(features)[0]
        print(f"{key:<15}: {predicted}")
