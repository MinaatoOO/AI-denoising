import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

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

# Dossiers des données
clean_dir = 'DATA2/clean'
noisy_dir = 'DATA2/noisy'

# Charger les fichiers clean et noisy
clean_audio, clean_files = load_audio_files(clean_dir)
noisy_audio, noisy_files = load_audio_files(noisy_dir)

print(clean_files==noisy_files) #verifier si data2 d'alexy est bien coherente lawolakahba

# Construire les spectrogrammes
def create_spectrograms(audio_data, n_fft=1024, hop_length=512):
    spectrograms = []
    for y, sr in audio_data:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        spectrograms.append(S)
    return spectrograms

#notre liste des spectrogrammes
clean_spectrograms = create_spectrograms(clean_audio)
noisy_spectrograms = create_spectrograms(noisy_audio)

# Normalisation et sauvegarde des paramètres de normalisation
def normalize_spectrograms(spectrograms):
    normalized_spectrograms = []
    normalization_params = []  # Liste pour sauvegarder les valeurs max
    for S in spectrograms:
        max_value = np.max(S)
        normalization_params.append(max_value)
        normalized_spectrograms.append(S / max_value)
    return np.array(normalized_spectrograms), normalization_params

normalized_clean_spectrograms, clean_norm_params = normalize_spectrograms(clean_spectrograms)
normalized_noisy_spectrograms, noisy_norm_params = normalize_spectrograms(noisy_spectrograms)

# Créer X(noisy) et Y(clean)
X = np.array(normalized_noisy_spectrograms)  # Input: données bruitées
Y = np.array(normalized_clean_spectrograms) # Output: données propres


# Diviser X et Y (spectrogrammes) pour entraînement et test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Diviser noisy_files (noms des fichiers) en ensemble d'entraînement et de test
noisy_train_files, noisy_test_files = train_test_split(noisy_files, test_size=0.2, random_state=42)

print(noisy_test_files)

# Restructurer les données pour traiter colonne par colonne
def prepare_column_data(X, Y):
    n_samples, n_freq, n_time = X.shape
    X_columns = X.transpose(0, 2, 1).reshape(-1, n_freq)  # (n_samples * n_time, n_freq)
    Y_columns = Y.transpose(0, 2, 1).reshape(-1, n_freq)  # (n_samples * n_time, n_freq)
    return X_columns, Y_columns

X_train_columns, Y_train_columns = prepare_column_data(X_train, Y_train)
X_test_columns, Y_test_columns = prepare_column_data(X_test, Y_test)

# Sauvegarde et chargement du modèle
model_path = "mlp_denoising_model.h5"


model = Sequential([
Dense(1024, activation='relu', input_shape=(X_train_columns.shape[1],)),
Dense(512, activation='relu'),
Dense(256, activation='relu'),
Dense(128, activation='relu'),
Dense(Y_train_columns.shape[1], activation='linear')
])


    # Compiler le modèle
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Entraîner le modèle
history = model.fit(
    X_train_columns, Y_train_columns,
    validation_data=(X_test_columns, Y_test_columns),
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
loss, mae = model.evaluate(X_test_columns, Y_test_columns)
print(f"evaluation finale du modele : Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")















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
