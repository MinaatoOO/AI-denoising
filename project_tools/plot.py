"""
Nom du Fichier : plot.py
Description    : regroupe les fonctions d'affichage d'un fichier .FLAC
Auteur         : Bounsavath
Date           : 13 11 2024
Version        : 1.0
"""

import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import Spectrogram

def plot_waveform(file_path):
    # Charger le fichier audio
    waveform, sample_rate = torchaudio.load(file_path)

    # Calculer le temps correspondant à chaque échantillon
    num_samples = waveform.shape[1]
    time = np.linspace(0, num_samples / sample_rate, num_samples)

    # Tracer la forme d'onde
    plt.figure(figsize=(10, 4))
    plt.plot(time, waveform.t().numpy())
    plt.title("Forme d'onde du fichier audio")
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()


def plot_spect(file_path, n_fft=1024, hop_length=None):
    # Charger le fichier audio
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Calculer le spectrogramme
    spectrogram_transform = Spectrogram(n_fft=n_fft, hop_length=hop_length)
    spectrogram = spectrogram_transform(waveform)
    
    # Convertir le spectrogramme en dB (en utilisant log naturel pour l'illustration)
    spectrogram_db = 10 * np.log(spectrogram[0].numpy() + 1e-10)
    
    # Tracer le spectrogramme
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_db, aspect='auto', origin='lower', 
               extent=[0, waveform.shape[1] / sample_rate, 0, sample_rate / 2])
    plt.title("Spectrogramme")
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Fréquence (Hz)")
    plt.colorbar(label="Amplitude (dB)")
    plt.show()

def plot_melspect(file_path, n_fft=1024, hop_length=None, n_mels=64):
    # Charger le fichier audio
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Calculer le Mel spectrogramme
    mel_spectrogram_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spectrogram = mel_spectrogram_transform(waveform)
    
    # Convertir le Mel spectrogramme en dB pour la visualisation
    mel_spectrogram_db = 10 * np.log(mel_spectrogram[0].numpy() + 1e-10)
    
    # Tracer le Mel spectrogramme
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram_db, aspect='auto', origin='lower', 
               extent=[0, waveform.shape[1] / sample_rate, 0, sample_rate / 2])
    plt.title("Mel Spectrogramme")
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Fréquence Mel")
    plt.colorbar(label="Amplitude (dB)")
    plt.show()
