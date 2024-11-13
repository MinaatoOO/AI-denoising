"""
Nom du Fichier : MLP.py
Description    : à pour but de tester les afficheurs d'un fichier .FLAC
Auteur         : Bounsavath
Date           : 13 11 2024
Version        : 1.0
"""
import pickle
from project_tools.compute import *
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_squared_error

##CONFIG
n_fft=1024
do_train = 1

##RUN
path_clean = "DATA/train/clean"
path_noisy = "DATA/train/noisy"
path_clean_test = "DATA/test/clean"
path_noisy_test = "DATA/test/noisy"


clean_tensor = process_flac_files(path_clean,n_fft)
noisy_tensor = process_flac_files(path_noisy,n_fft)
noisy_test_tensor = process_flac_files(path_noisy_test,n_fft)
clean_test_tensor = process_flac_files(path_clean_test,n_fft)

if do_train:
    #initialisation du modèle
    mlp = MLPRegressor(
        hidden_layer_sizes=(64),  # Taille de la couche cachée (ici, une seule couche de 128 neurones)
        activation='relu',          # Fonction d'activation
        solver='adam',              # Optimiseur
        max_iter=200,                # Nombre maximum d'itérations
        random_state=42
    )

    #entrainement
    mlp.fit(noisy_tensor,clean_tensor)
    #on enregistre le modèle
    pickle.dump(mlp,open('modele/mlp.dat','wb'))


#test
mlp = pickle.load(open('modele/mlp.dat', 'rb'))
predicted_tensor = mlp.predict(noisy_test_tensor)
mse = mean_squared_error(clean_test_tensor, predicted_tensor)
print(mse)

#mise en forme des résulats

predicted_tensor = predicted_tensor.T
# Spécifier un répertoire de sauvegarde
predicted_spect_folder = "Result/predicted_spect/"
# Vérifier si le répertoire existe, sinon le créer
if not os.path.exists(predicted_spect_folder):
    os.makedirs(predicted_spect_folder)

#récupération des noms originaux: 
noisy_test_filenames = os.listdir(path_noisy_test)
segment_height = predicted_tensor.shape[0]
segment_width = int(predicted_tensor.shape[1]/len(noisy_test_filenames))
nombre_segment = int(predicted_tensor.shape[1]/segment_width)
for i in range(nombre_segment):
    segment = predicted_tensor[:,i * segment_width:(i + 1) * segment_width]
    filename = noisy_test_filenames[i]
    filename = filename.replace(".flac","")
    
    # Sauvegarder chaque segment dans le répertoire spécifié
    np.save(os.path.join(predicted_spect_folder, filename), segment)


# Spécifier un répertoire de sauvegarde
predicted_flac_folder = "Result/predicted_flac/"
# Vérifier si le répertoire existe, sinon le créer
if not os.path.exists(predicted_flac_folder):
    os.makedirs(predicted_flac_folder)

spect_to_flac(predicted_spect_folder,predicted_flac_folder)

