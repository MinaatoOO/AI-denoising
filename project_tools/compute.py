"""
Nom du Fichier : compute.py
Description    : regroupe les fcts qui applique des calculs à un fichier .flac
Auteur         : Bounsavath
Date           : 13 11 2024
Version        : 1.0
"""

import torchaudio
import os
import torch
import numpy as np

def compute_spectrogram(waveform, n_fft):
    # Transformer un signal audio en spectrogramme
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,          # Nombre de points pour la FFT
        win_length=None,     # Longueur de la fenêtre (None pour utiliser n_fft)
        hop_length=512,       # Décalage de la fenêtre
        power=2               #on passe direct au omdule au carré
    )
    spectrogram = spectrogram_transform(waveform)
    return spectrogram

def process_flac_files(folder_path,n_fft=1024):
    spectrograms = []  # Liste vide pour stocker les spectrogrammes

    for filename in os.listdir(folder_path):
        if filename.endswith(".flac"):
            # Charger le fichier audio
            path = os.path.join(folder_path, filename)
            waveform, sample_rate = torchaudio.load(path)
            
            # Calculer le spectrogramme
            spectrogram = compute_spectrogram(waveform, n_fft)
            
            # Ajouter le spectrogramme à la liste
            spectrograms.append(spectrogram)

    # Concaténer tous les spectrogrammes dans une matrice 2D
    
    # On les concatène sur la dimension 1 (les colonnes)
    spectrograms_tensor = torch.cat(spectrograms, dim=2)  # Concaténation horizontale
    ajusted = spectrograms_tensor.squeeze().numpy()
    return ajusted.T

def spect_to_flac(spect_path,flac_path):
    filenames = sorted(os.listdir(spect_path))
    griffinlim_transform = torchaudio.transforms.GriffinLim(n_fft=1024, win_length=None, hop_length=512)
    for i in filenames:
        i=i.replace(".npy","")
        #chargement des fichiers qu'on a besoin donc le spectrogramme et 
        # le signal waveform originalpour dénormaliser
        spect = np.load(spect_path+i+".npy")
        original_waveform,fe = torchaudio.load('DATA/DataTotale/'+i+'.flac')

        #convertion du np array en tensor
        spect_tensor = torch.tensor(spect)

        #passage de spect a wf:
        waveform_normalized = griffinlim_transform(spect_tensor)

        #dénormalization
        waveform_unnormalized = waveform_normalized*torch.abs(torch.max(original_waveform))

        #enregistré le fichier
        torchaudio.save(os.path.join(flac_path, i+".flac"),waveform_unnormalized.unsqueeze(0).clone().detach(),fe, format = 'flac')
