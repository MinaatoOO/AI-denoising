import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Charger le modèle enregistré
model_path = "mlp_denoising_model_-20dB.h5"
model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
print(f"Modèle chargé depuis : {model_path}")

# Fonction pour charger les fichiers audio
def load_audio_files(data_dir):
    audio_data = []
    file_names = []
    for file in os.listdir(data_dir):
        if file.endswith('.flac'):
            file_path = os.path.join(data_dir, file)
            y, sr = librosa.load(file_path, sr=None)  # sr=None pour garder la fréquence d'échantillonnage originale
            audio_data.append((y, sr))
            file_names.append(file)
    return audio_data, file_names

# Dossiers des fichiers bruités, propres et totaux
clean_dir = 'Processed_Audio_-20dB/clean'
noisy_dir = 'Processed_Audio_-20dB/noisy'
total_dir = 'Processed_Audio_-20dB/noisy'

# Charger les fichiers clean, noisy et originaux
clean_audio, clean_files = load_audio_files(clean_dir)
noisy_audio, noisy_files = load_audio_files(noisy_dir)

random_indices = []
for i in range(len(clean_files)):
    if clean_files[i] in ['1060.flac','840.flac','1087.flac']:
        random_indices.append(i)


# Fonction pour créer des spectrogrammes (amplitude + phase)
def create_spectrogram_with_phase(y, n_fft=1024, hop_length=512):
    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(S_complex)
    phase = np.angle(S_complex)
    return magnitude, phase

# Fonction pour reconstruire l'audio avec amplitude et phase
def spectrogram_to_audio_with_phase(magnitude, phase, hop_length=512):
    S_complex = magnitude * np.exp(1j * phase)  # Recombinaison amplitude et phase,l'ecriture d'un nombre complexe
    return librosa.istft(S_complex, hop_length=hop_length)

# Répertoire de sauvegarde des fichiers prédits
output_dir = os.path.join('Processed_Audio_-20dB', 'predicted_audio_-20dB')
os.makedirs(output_dir, exist_ok=True)  # Crée le dossier s'il n'existe pas , il va se créer au debut

# Sélectionner 3 fichiers de test sur les quels le modele n'est pas encore entrainé
#random_indices = [616,1270,499]
for idx in random_indices:
    # Charger un fichier bruité et son équivalent propre
    noisy, sr = noisy_audio[idx]
    clean, _ = clean_audio[idx]
    file_name = noisy_files[idx]

    print(f"Test sur le fichier : {file_name}")

    # Charger l'audio complet pour récupérer la phase
    total_audio_path = os.path.join(total_dir, file_name)
    total_audio, _ = librosa.load(total_audio_path, sr=None)

    # Créer le spectrogramme bruité et propre
    noisy_spectrogram, noisy_phase = create_spectrogram_with_phase(noisy)
    clean_spectrogram, phasee = create_spectrogram_with_phase(clean)
    _, original_phase = create_spectrogram_with_phase(total_audio) #phase du signale bruité

    # Normaliser le spectrogramme bruité pour correspondre au modèle
    max_value = np.max(noisy_spectrogram)
    normalized_noisy_spectrogram = noisy_spectrogram / max_value

    # Diviser le spectrogramme en colonnes pour le modèle
    noisy_columns = normalized_noisy_spectrogram.T  # Chaque colonne est une tranche temporelle pour correspondre au modeles
    predicted_columns = model.predict(noisy_columns)

    # Reconstruire le spectrogramme prédit
    predicted_spectrogram = predicted_columns.T * max_value  # Dénormaliser

    # Utiliser la phase originale pour reconstruire l'audio
    audio_noisy = spectrogram_to_audio_with_phase(noisy_spectrogram, noisy_phase)
    audio_clean = spectrogram_to_audio_with_phase(clean_spectrogram, noisy_phase)  # Pour cohérence
    audio_predicted = spectrogram_to_audio_with_phase(predicted_spectrogram, phasee)

    # Enregistrer le fichier audio débruité
    output_file = os.path.join(output_dir, f"predicted_{file_name.split('.')[0]}.wav")
    sf.write(output_file, audio_predicted, sr)
    print(f"Fichier audio débruité enregistré : {output_file}")

    # Tracer les spectrogrammes
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(noisy_spectrogram, ref=np.max), sr=sr, hop_length=512, y_axis='log', x_axis='time')
    plt.title("Spectrogramme Bruité (Noisy)_-20dB")
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(clean_spectrogram, ref=np.max), sr=sr, hop_length=512, y_axis='log', x_axis='time')
    plt.title("Spectrogramme Propre (Clean)")
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(predicted_spectrogram, ref=np.max), sr=sr, hop_length=512, y_axis='log', x_axis='time')
    plt.title("Spectrogramme Débruité (Predicted)_-20dB")
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()

    # Tracer les formes d'onde
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.title("Forme d'onde Bruité (Noisy)")
    plt.plot(audio_noisy)

    plt.subplot(3, 1, 2)
    plt.title("Forme d'onde Propre (Clean)")
    plt.plot(audio_clean)

    plt.subplot(3, 1, 3)
    plt.title("Forme d'onde Débruitée (Predicted)")
    plt.plot(audio_predicted)

    plt.tight_layout()
    plt.show()
