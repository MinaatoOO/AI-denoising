import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt

# Paramètres de transformation pour le spectrogramme
sample_rate = 16000
n_fft = 512
hop_length = 256
spectrogram_transform = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)
db_transform = transforms.AmplitudeToDB()

# Dataset personnalisé avec option d'utiliser un spectrogramme en valeurs absolues ou en dB
class DenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, use_db=True):
        self.clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith(".flac")])
        self.noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith(".flac")])
        self.use_db = use_db  # Choisir d'utiliser l'échelle dB ou non

        # Calculer la taille réelle du spectrogramme aplati
        sample_waveform, _ = torchaudio.load(self.clean_files[0])
        sample_spectrogram = spectrogram_transform(sample_waveform).squeeze().abs()
        if self.use_db:
            sample_spectrogram = db_transform(sample_spectrogram)  # Appliquer l'échelle dB si nécessaire
        self.input_size = sample_spectrogram.numel()

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Charger le signal clean et bruité
        clean_waveform, _ = torchaudio.load(self.clean_files[idx])
        noisy_waveform, _ = torchaudio.load(self.noisy_files[idx])
        
        # Calculer les spectrogrammes
        clean_spectrogram = spectrogram_transform(clean_waveform).squeeze().abs()
        noisy_spectrogram = spectrogram_transform(noisy_waveform).squeeze().abs()
        
        # Appliquer la transformation en dB si spécifié
        if self.use_db:
            clean_spectrogram = db_transform(clean_spectrogram)
            noisy_spectrogram = db_transform(noisy_spectrogram)
        
        # Normaliser les spectrogrammes
        clean_spectrogram = (clean_spectrogram - clean_spectrogram.mean()) / clean_spectrogram.std()
        noisy_spectrogram = (noisy_spectrogram - noisy_spectrogram.mean()) / noisy_spectrogram.std()
        
        # Aplatir les spectrogrammes pour les passer dans le MLP
        clean_spectrogram = clean_spectrogram.reshape(-1)
        noisy_spectrogram = noisy_spectrogram.reshape(-1)
        
        return noisy_spectrogram, clean_spectrogram

# Préparation des données avec l'option d'utiliser un spectrogramme en dB ou non
clean_dir = "DATA/train/clean"
noisy_dir = "DATA/train/noisy"
use_db = True  # Mettez False pour utiliser les valeurs absolues
train_dataset = DenoisingDataset(clean_dir, noisy_dir, use_db=use_db)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# Taille d'entrée pour le MLP
input_size = train_dataset.input_size
print(f"Taille d'entrée calculée pour le MLP (spectrogramme aplati) : {input_size}")

# Définition de l'architecture du MLP pour le débruitage
class DenoisingMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DenoisingMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

# Initialisation du modèle et de l'optimiseur
model = DenoisingMLP(input_size, 1024, input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Fonction d'entraînement avec enregistrement de la perte pour chaque époque
def train(model, dataloader, criterion, optimizer, scheduler, epochs=15):
    model.train()
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for noisy_spectrogram, clean_spectrogram in dataloader:
            noisy_spectrogram, clean_spectrogram = noisy_spectrogram.float(), clean_spectrogram.float()
            optimizer.zero_grad()
            outputs = model(noisy_spectrogram)
            loss = criterion(outputs, clean_spectrogram)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    return losses

# Entraînement du modèle
losses = train(model, train_loader, criterion, optimizer, scheduler, epochs=15)

# Sauvegarder le modèle entraîné
torch.save(model.state_dict(), "denoising_mlp_model.pth")
print("Modèle sauvegardé sous 'denoising_mlp_model.pth'")

# Affichage de la courbe de perte
plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-')
plt.xlabel("Époques")
plt.ylabel("Perte (MSE)")
plt.title("Variation de la perte au fil des époques")
plt.grid()
plt.show()
