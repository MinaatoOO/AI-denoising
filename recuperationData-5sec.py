import os
import torchaudio

# Chemin vers le dossier LibriSpeech
root_dir = 'LibriSpeech/dev-clean'
output_dir = 'DATA/DataTotale'


# Variables pour compter le nombre de fichiers sauvegardés et la durée totale
file_count = 0
total_duration = 0  # ici en secondes

# Fonction pour extraire les 5 premières secondes d'un fichier audio
def extract_first_5_seconds(input_path, output_path):
    audio, sample_rate = torchaudio.load(input_path) # load le fichier , audio est un tenseur pytorch
    segment_duration = 5 * sample_rate  # nombre d'echantillons present dans 5 secondes , generalement Te=N/T

    # Prend les 5 premières secondes si l'audio est plus long
    if audio.shape[1] >= segment_duration:
        segment = audio[:, :segment_duration]
        torchaudio.save(output_path, segment, sample_rate, format="FLAC")
        return 5  # Retourne la durée en secondes du segment sauvegardé
    return 0

# Parcourt tous les sous-dossiers et fichiers FLAC dans le dossier dev-clean
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.flac'):
            file_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, f"{file_count}.flac")

            # Extrait les 5 premières secondes et met à jour le nombre de fichiers et la durée
            duration = extract_first_5_seconds(file_path, output_path)
            if duration > 0:
                file_count += 1
                total_duration += duration

# Conversion de la durée totale en heures
total_hours = total_duration / 3600

print(f"Nombre de fichiers sauvegardés : {file_count}")
print(f"Durée totale des fichiers sauvegardés : {total_hours:.2f} heures")
