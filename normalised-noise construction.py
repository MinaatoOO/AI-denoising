import matplotlib.pyplot as plt
import os
import torchaudio
import torch
import numpy as np
from tqdm import tqdm  # Importer tqdm pour la barre de progression




def normalize_audio(file_path, output_path):
    """
    Normalise l'amplitude d'un fichier audio pour qu'elle soit entre -1 et 1.
    """
    # Charger le fichier audio
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Calculer la valeur maximale absolue de l'amplitude
    max_amplitude = waveform.abs().max()
    
    # Diviser tout le signal par cette valeur maximale pour normaliser entre -1 et 1
    normalized_waveform = waveform / max_amplitude
    
    # Sauvegarder le fichier normalisé
    torchaudio.save(output_path, normalized_waveform, sample_rate)

# Répertoires des données
data_dir = "DATA"  # Répertoire contenant train/test et leurs sous-dossiers

# Boucle à travers les dossiers train/clean et test/clean pour normaliser les fichiers
for dataset_type in ["train", "test"]:
    clean_dir = os.path.join(data_dir, dataset_type, "clean")
    
    for filename in os.listdir(clean_dir):
        if filename.endswith(".flac"):
            file_path = os.path.join(clean_dir, filename)
            output_path = file_path  # Remplacer le fichier original
            
            # Normaliser le fichier audio
            normalize_audio(file_path, output_path)




'''
data_dir='DATA/DataTotale'
filles=os.listdir(data_dir)

s, Fe = torchaudio.load(os.path.join(data_dir,filles[0]))
S, Fee = torchaudio.load('babble_16k.wav')
print(Fee==Fe)
# Convertir le tenseur en tableau NumPy pour l'afficher
waveform = s.numpy()[0]  # Puisque le fichier est mono, s[0] est l'unique canal
time = torch.arange(0, waveform.shape[0]) / Fe  # Créer l'axe du temps en secondes

# Tracer la forme d'onde
plt.figure(figsize=(12, 4))
plt.plot(time, waveform)
plt.title("Forme d'onde audio")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.show()
'''



import random

def calculate_alpha(p_signal, rsb_db, b):
    """
    Calcule le facteur alpha pour ajuster le bruit à la puissance spécifiée du RSB.
    """
    # Calculer la puissance cible du bruit
    p_bruit = p_signal / (10 ** (rsb_db / 10))
    # Calculer le facteur alpha
    alpha = np.sqrt(p_bruit / torch.mean(b ** 2).item())
    return alpha

def add_noise(clean_path, noise, rsb_db, output_path, sample_rate):
    """
    Charge un fichier clean, extrait un segment aléatoire du bruit, ajoute le bruit avec le RSB spécifié, et sauvegarde le fichier bruité.
    """
    # Charger le fichier clean
    s, _ = torchaudio.load(clean_path)
    num_samples = s.shape[1]  # Nombre d'échantillons pour 5 secondes
    
    # Sélectionner un segment aléatoire de 5 secondes du bruit
    max_start = noise.shape[1] - num_samples  # Limite pour éviter de dépasser la fin du bruit , noise.shape[1] retourne le nombre d'echatillons dans le bruit
    start = random.randint(0, max_start)
    noise_segment = noise[:, start:start + num_samples] # la preumier : signifie qu'on prenne tout les cannaux , apres le debut et la fin du segment
    
    # Calculer la puissance du signal clean
    p_signal = torch.mean(s ** 2).item()
    
    # Calculer alpha pour ajuster le RSB
    alpha = calculate_alpha(p_signal, rsb_db, noise_segment)
    
    # Créer le signal bruité
    noisy_signal = s + alpha * noise_segment
    
    # Sauvegarder le fichier bruité avec le même nom que le fichier clean
    torchaudio.save(output_path, noisy_signal, sample_rate)

# Paramètres
rsb_db = 10  # Définir le RSB souhaité en dB
noise_path = "babble_16k.wav"  # Chemin vers le fichier de bruit
data_dir = "DATA"  # Répertoire contenant train/test et leurs sous-dossiers

# Charger le bruit une seule fois
noise, sample_rate = torchaudio.load(noise_path)

max= noise.abs().max()
    
# Diviser tout le signal par cette valeur maximale pour normaliser entre -1 et 1
noise = noise / max

# Boucle à travers les fichiers clean pour ajouter le bruit
for dataset_type in ["train", "test"]:
    clean_dir = os.path.join(data_dir, dataset_type, "clean")
    noisy_dir = os.path.join(data_dir, dataset_type, "noisy")
    
    # Obtenir la liste des fichiers .flac et envelopper dans tqdm pour la barre de progression
    file_list = [f for f in os.listdir(clean_dir) if f.endswith(".flac")]
    
    for filename in tqdm(file_list, desc=f"Traitement des fichiers {dataset_type}", unit="fichier"):
        clean_path = os.path.join(clean_dir, filename)
        noisy_path = os.path.join(noisy_dir, filename)  # Conserver le même nom pour le fichier bruité
        
        # Appliquer le bruit avec un segment aléatoire et le RSB spécifié
        add_noise(clean_path, noise, rsb_db, noisy_path, sample_rate)




#ici on normalise le noisy
for dataset_type in ["train", "test"]:
    noisyy_dir = os.path.join(data_dir, dataset_type, "noisy")
    
    for filename in os.listdir(noisyy_dir):
        if filename.endswith(".flac"):
            noise_path = os.path.join(noisyy_dir, filename)
            noisee,Feee=torchaudio.load(noise_path)  # charger le fichier bruité
            # Normaliser le fichier audio bruité
            maxi=noisee.abs().max()
            noisee=noisee/maxi
                    


