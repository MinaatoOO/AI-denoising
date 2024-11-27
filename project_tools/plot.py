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


def plot_spect(file_path, n_fft=512, hop_length=256):
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

def plot_melspect(file_path, n_fft=512, hop_length=256, n_mels = 128):
    # Charger le fichier audio
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Calculer le spectrogramme
    melspectrogram_transform = Spectrogram(
        ample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
        )
    melspectrogram = melspectrogram_transform(waveform)
    
    # Convertir le spectrogramme en dB (en utilisant log naturel pour l'illustration)
    spectrogram_db = 10 * np.log(melspectrogram[0].numpy() + 1e-10)
    
    # Tracer le spectrogramme
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_db)
    plt.title("Spectrogramme")
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Fréquence (Hz)")
    plt.colorbar(label="Amplitude (dB)")
    plt.show()

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def plot_melspect(file_path, n_fft=512, hop_length=256, n_mels=128):
    """
    Calculer et afficher le melspectrogramme en dB d'un fichier audio .flac
    avec des fréquences en Hz sur l'axe des ordonnées en utilisant librosa.
    
    :param file_path: Chemin du fichier audio .flac
    :param n_fft: Taille de la fenêtre FFT
    :param hop_length: Longueur du saut entre fenêtres
    :param n_mels: Nombre de bandes Mel
    """
    # Charger le fichier audio
    waveform, sample_rate = librosa.load(file_path, sr=None)  # sr=None pour conserver la fréquence d'origine
    
    # Calculer le melspectrogramme
    melspectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convertir l'amplitude en décibels
    melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)
    
    # Tracer le melspectrogramme
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        melspectrogram_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='log'
    )
    plt.title("Melspectrogramme (Fréquences en Hz)")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.show()


