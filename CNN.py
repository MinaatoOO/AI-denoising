import os
import librosa
import numpy as np
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Resizing
import matplotlib.pyplot as plt
import pickle

train=0
test=1

output_dir = "./Result_CNN/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = "cnn_denoising_model.h5"


# Charger les fichiers audio
def load_audio_files(data_dir):
    audio_data = []
    file_names = []  # Sauvegarder les noms des fichiers pour la phase de test
    for file in os.listdir(data_dir):
        if file.endswith('.flac'):
            file_path = os.path.join(data_dir, file)
            y, sr = librosa.load(file_path, sr=None)
            audio_data.append((y, sr))
            file_names.append(file)
    return audio_data, file_names

# Construire les spectrogrammes
def create_spectrograms(audio_data, sr=16000, n_fft=1024, hop_length=512):
    num_frames = math.ceil((5 * sr - hop_length) / (n_fft - hop_length))  # Approximation
    spectrograms = np.zeros((len(audio_data), int(n_fft / 2) + 1, num_frames, 1))

    for i, y in enumerate(audio_data):
        # Calcul du spectrogramme pour chaque signal
        S = np.abs(librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length))
        # Vérification des dimensions
        if S.shape[1] > num_frames:
            S = S[:, :num_frames]  # Ajuste à la taille attendue
        spectrograms[i, :, :S.shape[1], 0] = S  # Ajouter à la sortie

    return spectrograms


# Normalisation et sauvegarde des paramètres de normalisation
def normalize_spectrograms(spectrograms):
    normalized_spectrograms = np.zeros_like(spectrograms)
    normalization_params = np.zeros(len(spectrograms))  # Liste pour sauvegarder les valeurs max
    for i, S in enumerate(spectrograms):
        max_value = np.max(S)
        normalization_params[i]= max_value
        normalized_spectrograms[i] = (S / max_value)
    return normalized_spectrograms, normalization_params




# Dossiers des données
clean_dir = 'DATA2/clean'
noisy_dir = 'DATA2/noisy'

# Charger les fichiers clean et noisy
clean_audio, clean_files = load_audio_files(clean_dir)
noisy_audio, noisy_files = load_audio_files(noisy_dir)

print(clean_files==noisy_files) #verifier si data2 d'alexy est bien coherente lawolakahba



#notre liste des spectrogrammes
clean_spectrograms = create_spectrograms(clean_audio)
noisy_spectrograms = create_spectrograms(noisy_audio)



normalized_clean_spectrograms, clean_norm_params = normalize_spectrograms(clean_spectrograms)
normalized_noisy_spectrograms, noisy_norm_params = normalize_spectrograms(noisy_spectrograms)

# Créer X(noisy) et Y(clean)
X = normalized_noisy_spectrograms   # Input: données bruitées
Y = normalized_clean_spectrograms   # Output: données propres


# Diviser X et Y (spectrogrammes) pour entraînement et test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

np.save(output_dir + 'X_train.npy', X_train)
np.save(output_dir + 'X_test.npy', X_test)
np.save(output_dir + 'Y_train.npy', Y_train)
np.save(output_dir + 'Y_test.npy',Y_test)
np.save(output_dir + 'noisy_norm_params.npy', noisy_norm_params)

# Diviser noisy_files (noms des fichiers) en ensemble d'entraînement et de test
noisy_train_files, noisy_test_files = train_test_split(noisy_files, test_size=0.2, random_state=42)

print(noisy_test_files)

# Sauvegarde et chargement du modèle
model_path = "cnn_denoising_model.h5"
input_shape = X[0].shape

model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2), padding='same'),#shape 257x78
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),#shape 129x39
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),#shape 65x20
        # Les couches de convolution transposée (deconvolution) pour la reconstruction de l'image
        Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides = (2,2)),#129x39
        Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides = (2,2)),#257x78
        Conv2DTranspose(1, (3, 3), activation='relu', padding='same', strides = (2,2)),
        Resizing(513, 156)
        # La couche de sortie pour générer l'image
        #Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Sortie avec une image de même taille
    ])


    # Compiler le modèle
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Entraîner le modèle
history = model.fit(
X_train, Y_train,
validation_data=(X_test, Y_test),
epochs=10,
batch_size=64
)

    # Sauvegarder le modèle après entraînement
model.save(model_path)
print(f"Modèle sauvegardé dans : {model_path}")

    # Tracer les erreurs
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Évaluer le modèle
loss, mae = model.evaluate(X_test, Y_test)
print(f"evaluation finale du modele : Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")


# Charger le modèle enregistré
model_path = "mlp_denoising_model.h5"
model = load_model(model_path)
print(f"Modèle chargé depuis : {model_path}")

#charger les variables qui sont deja normalisée
X_test = np.load(output_dir + '/X_test.npy')
Y_test = np.load(output_dir + '/Y_test.npy')
noisy_norm_params = np.load(output_dir + 'noisy_norm_params.npy')

#prediction sur x_test
Y_predicted = model.predict(X_test)

#denormalisation
for i, max in enumerate(noisy_norm_params):
    Y_predicted[i] = Y_predicted[i]*max[i]















# --- Partie test : prédiction sur un fichier audio ---
'''
# Choisir un fichier de test
test_index = np.random.randint(len(X_test))
test_spectrogram = X_test[test_index]  # Spectrogramme bruité
test_clean_spectrogram = Y_test[test_index]  # Spectrogramme propre correspondant
test_file_name = noisy_test_files[test_index]

print(f"Test sur le fichier : {test_file_name}")

# Restructurer le spectrogramme bruité pour le modèle
test_columns = test_spectrogram.T  # Chaque colonne est une tranche temporelle
predicted_columns = model.predict(test_columns)

# Reconstruire le spectrogramme prédit
predicted_spectrogram = predicted_columns.T

# --- Reconstruire les fichiers audio ---
def spectrogram_to_audio(spectrogram, hop_length=256):
    return librosa.istft(spectrogram, hop_length=hop_length)

audio_noisy = spectrogram_to_audio(test_spectrogram)
audio_clean = spectrogram_to_audio(test_clean_spectrogram)
audio_predicted = spectrogram_to_audio(predicted_spectrogram)

# --- Afficher les formes d'onde ---
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.title("Signal bruité (Noisy)")
plt.plot(audio_noisy)
plt.subplot(3, 1, 2)
plt.title("Signal propre (Clean)")
plt.plot(audio_clean)
plt.subplot(3, 1, 3)
plt.title("Signal prédit (Predicted)")
plt.plot(audio_predicted)

plt.tight_layout()
plt.show()

# Écouter les fichiers audio
librosa.output.write_wav("noisy_test.wav", audio_noisy, sr=16000)
librosa.output.write_wav("clean_test.wav", audio_clean, sr=16000)
librosa.output.write_wav("predicted_test.wav", audio_predicted, sr=16000)

print("Fichiers audio générés : noisy_test.wav, clean_test.wav, predicted_test.wav")

'''
