"""
Nom du Fichier : test_plot.py
Description    : à pour but de tester les afficheurs d'un fichier .FLAC
Auteur         : Bounsavath
Date           : 13 11 2024
Version        : 1.0
"""
from project_tools.plot import *
from project_tools.compute import *
import matplotlib.pyplot as plt
import os
import torchaudio
import torch
import numpy as np
from tqdm import tqdm  # Importer tqdm pour la barre de progression
import random

# path = "DATA/train/noisy/10.flac"
# wf,fe = torchaudio.load(path)
# # plot_waveform(path)
# plot_spect(path)

# # plot_melspect(path)
# a = compute_spectrogram(wf,1024)
# print(a.shape)


# Exemple de spectrogramme (remplacez-le par votre propre tableau NumPy)
# Supposons que `spectrogram` soit votre tableau NumPy contenant le spectrogramme.
# spectrogram_predicted = "Result/flac_synth/0_noisy_anasyn.flac"
# spectrogram_original = "Result/flac_synth/0_synth.flac"


# # Affichage du spectrogramme
# # plt.figure(figsize=(10, 4))
# # plt.imshow(spectrogram_predicted, aspect='auto', origin='lower', cmap='viridis')
# # plt.colorbar(format='%+2.0f dB')
# # plt.title("Spectrogram")
# # plt.xlabel("Time")
# # plt.ylabel("Frequency")
# # plt.show()

# # plot_spect(spectrogram_original)
# # plot_spect(spectrogram_predicted)

# # plot_melspect(spectrogram_original)
# # plot_melspect(spectrogram_predicted)

# plot_melspect(spectrogram_original)

n_fft = 512
win_length = 512
hop_length = 256 
window = 'hann'
audio_fs = 16000
fmin = 20
fmax = 8000
n_mels = 64


son0_clean, sr = librosa.load('Data2/clean/0.flac')
son0_noisy, sr = librosa.load('Data2/noisy/0.flac')

y_stft = librosa.stft(son0_clean,n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
Py = np.abs(y_stft)**2
S = librosa.feature.melspectrogram(S=Py, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax)

fig, axes = plt.subplots(2,1,figsize = (20,10))
img0 = librosa.display.specshow(S, sr=sr, hop_length=256, x_axis='time', y_axis='log', cmap='magma',ax=axes[0])
img1 = librosa.display.specshow(Py, sr=sr, hop_length=256, x_axis='time', y_axis='log', cmap='magma',ax=axes[1])
plt.title('Spectrogramme (échelle logarithmique)')
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")
plt.show()


