import torch
import torchaudio
import torchaudio.transforms as transforms
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Définir les paramètres de transformation (identiques à ceux utilisés lors de l'entraînement)
n_fft = 400
hop_length = 200
spectrogram_transform = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)
db_transform = transforms.AmplitudeToDB()

# Recréer la classe du modèle pour l'initialiser
class DenoisingMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DenoisingMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

# Taille d'entrée du modèle (devrait être calculée comme lors de l'entraînement)
input_size = 80601  # Vous devez ajuster cela en fonction de votre entrée réelle

# Initialiser le modèle avec la taille d'entrée correcte
model = DenoisingMLP(input_size, 1024, input_size)

# Charger les poids du modèle
model.load_state_dict(torch.load('denoising_mlp_model.pth'))
model.eval()  # Mettre le modèle en mode évaluation

# Charger un fichier audio à traiter (exemple)
audio_path_pred = 'DATA/train/noisy/0.flac'
noisy_pred, sample_rate = torchaudio.load(audio_path_pred)

clean_path = 'DATA/train/clean/0.flac'
clean, sample_rate = torchaudio.load(clean_path)  # Utilisez clean_path ici pour charger l'audio propre

# Transformer l'audio en spectrogramme
noisy_spectrogram = spectrogram_transform(noisy_pred).squeeze().abs()
noisy_spectrogram = db_transform(noisy_spectrogram)  # Si vous utilisiez l'échelle en dB lors de l'entraînement

clean_spectrogram = spectrogram_transform(clean).squeeze().abs()
clean_spectrogram = db_transform(clean_spectrogram)  # Appliquer la même transformation en dB

# Normaliser le spectrogramme (comme lors de l'entraînement)
noisy_spectrogram = (noisy_spectrogram - noisy_spectrogram.mean()) / noisy_spectrogram.std()
noisy_spectrogram = noisy_spectrogram.reshape(-1)  # Aplatir le spectrogramme

clean_spectrogram = (clean_spectrogram - clean_spectrogram.mean()) / clean_spectrogram.std()

# Convertir en tensor float et effectuer une prédiction
noisy_spectrogram = noisy_spectrogram.float().unsqueeze(0)  # Ajouter une dimension pour le batch

# Prédiction
with torch.no_grad():  # Désactiver le calcul des gradients pour la prédiction
    denoised_spectrogram = model(noisy_spectrogram)

# Squeeze le spectrogramme débruité et le convertir en numpy
denoised_spectrogram = denoised_spectrogram.squeeze().cpu().numpy()

# Reshape pour avoir la forme 2D nécessaire pour l'affichage (ajustez la taille selon le spectrogramme)
denoised_spectrogram = denoised_spectrogram.reshape((n_fft // 2 + 1, -1))  # Ajustez ici
  # Par exemple, reshaper en fonction de la dimension du spectrogramme original
clean_spectrogram = clean_spectrogram.numpy().reshape(n_fft // 2 + 1, -1)  # De même, reshape le spectrogramme clean

# Création du plot avec 2 sous-graphiques côte à côte
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot du spectrogramme débruité avec librosa
axs[0].set_title("Spectrogramme débruité")
librosa.display.specshow(denoised_spectrogram, x_axis='time', y_axis='log', ax=axs[0])
axs[0].set_xlabel("Frames")
axs[0].set_ylabel("Fréquence (log)")

# Plot du spectrogramme propre avec librosa
axs[1].set_title("Spectrogramme propre")
librosa.display.specshow(clean_spectrogram, x_axis='time', y_axis='log', ax=axs[1])
axs[1].set_xlabel("Frames")
axs[1].set_ylabel("Fréquence (log)")

# Affichage
plt.tight_layout()
plt.show()

denoised_flac=librosa.griffinlim(denoised_spectrogram)
sf.write("DATA/teeeeest.flac",denoised_flac,sample_rate,'PCM_16')
