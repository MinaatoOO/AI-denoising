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
clean_dir = 'Processed_Audio_-20dB/clean'
noisy_dir = 'Processed_Audio_-20dB/noisy'

# Charger les fichiers clean et noisy
clean_audio, clean_files = load_audio_files(clean_dir)
noisy_audio, noisy_files = load_audio_files(noisy_dir)

#print(clean_files==noisy_files) #verifier si data2 d'alexy est bien coherente lawolakahba

# Construire les spectrogrammes
def create_spectrograms(audio_data, n_fft=1024, hop_length=512):
    spectrograms = []
    for y, sr in audio_data:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        spectrograms.append(S)
    return spectrograms

#notre liste des spectrogrammes : liste des matrices
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
noisy_train_files, noisy_test_files = train_test_split(noisy_files, test_size=0.1, random_state=42)

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
model_path = "mlp_denoising_model_-20dB.h5"


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
