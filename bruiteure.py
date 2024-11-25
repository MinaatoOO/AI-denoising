import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import random


def process_audio_with_noise(input_dir, rsb_db, noise_path):
    """
    Ajoute du bruit à des fichiers audio propres avec un RSB spécifié, et crée des sous-dossiers clean et noisy.

    :param input_dir: Chemin vers le répertoire contenant les fichiers audio propres.
    :param rsb_db: Rapport signal/bruit (RSB) souhaité en dB.
    :param noise_path: Chemin vers le fichier de bruit (format WAV).
    """
    # Charger le bruit une fois
    noise, sr_noise = librosa.load(noise_path, sr=None)
    noise = noise / np.max(np.abs(noise))  # Normaliser le bruit entre -1 et 1
    
    # Répertoires de sortie
    output_dir = os.path.join(os.getcwd(), f"Processed_Audio_{rsb_db}dB") #os.getcwd()=repertoire de travail
    clean_output_dir = os.path.join(output_dir, "clean")
    noisy_output_dir = os.path.join(output_dir, "noisy")
    os.makedirs(clean_output_dir, exist_ok=True)
    os.makedirs(noisy_output_dir, exist_ok=True)
    
    # Parcourir les fichiers du dossier d'entrée
    file_list = [f for f in os.listdir(input_dir) if  f.endswith(".flac")]
    
    for filename in tqdm(file_list, desc="Traitement des fichiers", unit="fichier"):
        # Charger le fichier propre
        clean_path = os.path.join(input_dir, filename)
        clean_signal, sr_clean = librosa.load(clean_path, sr=None)
        
        # Vérifier que les taux d'échantillonnage sont cohérents
        if sr_clean != sr_noise:
            raise ValueError(f"Les taux d'échantillonnage diffèrent : {sr_clean} (clean) vs {sr_noise} (bruit).")
        
        num_samples = len(clean_signal)  # Nombre d'échantillons dans le signal propre
        
        # Extraire un segment aléatoire de bruit de même taille
        max_start = len(noise) - num_samples #on depasse jmais les bords du bruit
        if max_start < 0:
            raise ValueError("Le fichier de bruit est plus court que les fichiers propres !!")
        
        start = random.randint(0, max_start)
        noise_segment = noise[start:start + num_samples]
        
        # Calculer la puissance du signal propre
        p_signal = np.mean(clean_signal ** 2)
        
        # Calculer la puissance du bruit pour le RSB
        p_bruit = p_signal / (10 ** (rsb_db / 10))
        alpha = np.sqrt(p_bruit / np.mean(noise_segment ** 2))
        
        # Ajouter le bruit avec le RSB spécifié
        noisy_signal = clean_signal + alpha * noise_segment
        
        # Normaliser les signaux entre -1 et 1
        clean_signal = clean_signal / np.max(np.abs(clean_signal))
        noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
        
        # Sauvegarder les fichiers normalisés
        clean_output_path = os.path.join(clean_output_dir, filename)
        noisy_output_path = os.path.join(noisy_output_dir, filename)
        
        sf.write(clean_output_path, clean_signal, sr_clean)
        sf.write(noisy_output_path, noisy_signal, sr_clean)


# TEST :

input_dir = "Data2/clean"  # Répertoire contenant les fichiers propres
noise_path = "babble_16k.wav"  # Fichier de bruit
rsb_db =5  # RSB souhaité en dB

process_audio_with_noise(input_dir, rsb_db, noise_path)
