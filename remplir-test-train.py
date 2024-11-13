import os
import shutil
import random

# Dossiers source et destination
data_dir = 'DATA/DataTotale'
train_dir = 'DATA/train/clean'
test_dir = 'DATA/test/clean'


# Récupère tous les fichiers dans le dossier DataTotale
all_files = [f for f in os.listdir(data_dir) ] #os.listdir renvoie une liste contenant 

# Mélange aléatoire des fichiers pour une répartition aléatoire
#random.shuffle(all_files)

# Calcule le nombre de fichiers pour chaque ensemble
train_size = int(0.8 * len(all_files))  # 80% pour l'entraînement
test_size=int(0.2 * len(all_files))#20% pour le test

# Répartit les fichiers
train_files = all_files[:train_size]
test_files = all_files[train_size:]

# Copie les fichiers d'entraînement
for file_name in train_files:
    src_path = os.path.join(data_dir, file_name)
    dest_path = os.path.join(train_dir, file_name)
    shutil.copy(src_path, dest_path)

# Copie les fichiers de test
for file_name in test_files:
    src_path = os.path.join(data_dir, file_name)
    dest_path = os.path.join(test_dir, file_name)
    shutil.copy(src_path, dest_path)

# Affiche le nombre de fichiers et d'heure dans chaque ensemble
print(f"Nombre de fichiers et d'heure dans le dossier d'entraînement : {len(train_files)} , {0.8*2.2}h ")
print(f"Nombre de fichiers et d'heure dans le dossier de test : {len(test_files)}, {0.2*2.2}h")
