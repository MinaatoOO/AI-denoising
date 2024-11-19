import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Paramètres de transformation pour le spectrogramme
sample_rate = 16000
n_fft = 400
hop_length = 200
spectrogram_transform = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)  # Utiliser power=2 pour l'échelle de magnitude

#inverse_spectrogram_transform = transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length)

# Dataset personnalisé sans utiliser l'échelle dB
class DenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir):
        self.clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith(".flac")])
        self.noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith(".flac")])

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_waveform, _ = torchaudio.load(self.clean_files[idx])
        noisy_waveform, _ = torchaudio.load(self.noisy_files[idx])
        
        clean_spectrogram = spectrogram_transform(clean_waveform).squeeze().abs()
        noisy_spectrogram = spectrogram_transform(noisy_waveform).squeeze().abs()
        
        # Normalisation des spectrogrammes
        clean_spectrogram = (clean_spectrogram - clean_spectrogram.mean()) / clean_spectrogram.std()
        noisy_spectrogram = (noisy_spectrogram - noisy_spectrogram.mean()) / noisy_spectrogram.std()
        
        # Aplatir les spectrogrammes pour les passer dans le MLP
        clean_spectrogram = clean_spectrogram.reshape(-1)
        noisy_spectrogram = noisy_spectrogram.reshape(-1)
        
        return noisy_spectrogram, clean_spectrogram

# Définition de l'architecture du MLP pour le débruitage
class DenoisingMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DenoisingMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

# Préparation des données de test
clean_dir = "DATA/test/clean"
noisy_dir = "DATA/test/noisy"
global_data_dir = "DATA/DataTotale"  # Répertoire contenant les fichiers audio originaux pour la dénormalisation

# Calcul automatique de la taille d'entrée à partir des données de test
sample_waveform, _ = torchaudio.load(os.path.join(clean_dir, os.listdir(clean_dir)[0]))
sample_spectrogram = spectrogram_transform(sample_waveform).squeeze().abs()
input_size = sample_spectrogram.numel()  # Calcul de la taille du spectrogramme aplati


# Chargement du modèle sauvegardé
model_path = "denoising_mlp_model.pth"
model = DenoisingMLP(input_size, 1024, input_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Préparation des données de test
test_dataset = DenoisingDataset(clean_dir, noisy_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


print('end')