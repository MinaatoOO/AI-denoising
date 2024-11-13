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
spectrogram_predicted = np.load("Result/predicted_spect/8.npy")
spectrogram_original = "DATA/test/clean/8.flac"

# Affichage du spectrogramme
plt.figure(figsize=(10, 4))
plt.imshow(spectrogram_predicted, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()

plot_spect(spectrogram_original)

