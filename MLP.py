"""
Nom du Fichier : MLP.py
Description    : à pour but de réaliser une regression via MLP (Multi Layer Perceptron)
                 pour débruiter le un signal de conversation.
Auteur         : Bounsavath
Date           : 13 11 2024
Version        : 1.0
"""
import soundfile as sf
import librosa
import librosa.display
import numpy as np
import scipy
import matplotlib.pyplot as plt
from os import listdir, mkdir, system
from os.path import join, isdir, basename, splitext
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from project_tools.compute import *
import glob
import re
import pickle 

import matplotlib.pyplot as plt;

import pdb # Debugger: utilisez pdb.set_trace() pour mettre un point d'arrêt

##CONFIG

# Audio feature extraction 
n_fft = 1024
win_length = 400 
hop_length = 160 
window = 'hann'
audio_fs = 16000
fmin = 20
fmax = 8000
n_mels = 40

# MLP-based regression
mlp_arch=(256,128,64,128,256)
activation_function='relu'  #fonction d'activation: 'relu' tanh' 'swish'
solver='adam'               # Optimiseur: 'adam' 'lbfgs' 
max_iter=200                #nb max d'itérations
seed=42                     #seed pour la randomisation des poids initiaux
alpha=0.001                #regularisation

# Step
step_extract_features = 0
step_train = 1
step_test = 1
step_synth = 1

# path
noisy_audio_dir = 'Data2/noisy'
clean_audio_dir = 'Data2/clean'
output_dir = "./Result"
nb_sentences = 1585
# path_clean = "DATA/train/clean"
# path_noisy = "DATA/train/noisy"
# path_clean_test = "DATA/test/clean"
# path_noisy_test = "DATA/test/noisy"

##RUN

# Subfunction
def listdir_fullpath(d):
    return [join(d, f) for f in listdir(d)]
##################################################################
def extract_features(input_dir, clean_dir):
    
    if isdir(clean_dir) is False:
        mkdir(clean_dir)

    all_audio_filenames = sorted(listdir_fullpath(input_dir))

    for f in range(np.shape(all_audio_filenames)[0]):
        y, sr = librosa.load(all_audio_filenames[f],sr=audio_fs)

        #On commence par calculer la STFT de y 
        y_stft = librosa.stft(y,n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window)
        #On prends le module au carré pour avoir le spectre de puissance
        Py = np.abs(y_stft)**2
        #On se place dans l'échelle melspectrogramme
        S = librosa.feature.melspectrogram(S=Py, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax)

        np.save(clean_dir + '/' + splitext(basename(all_audio_filenames[f]))[0] + '.npy',S.transpose())

###########################################################################

def do_train(ind):
    # Load training data 
    all_noisy_features_filenames = sorted(listdir_fullpath(output_dir + '/noisy_features'))
    all_clean_features_filenames = sorted(listdir_fullpath(output_dir + '/clean_features'))
    max_nb_frames = 1000000 
    X_train = np.zeros((max_nb_frames,n_mels))
    y_train = np.zeros((max_nb_frames,n_mels))
    iter = 0
    for f in ind:
        noisy_features = np.load(all_noisy_features_filenames[f])
        clean_features = np.load(all_clean_features_filenames[f])
        X_train[iter:iter+np.shape(noisy_features)[0],:] = noisy_features
        y_train[iter:iter+np.shape(noisy_features)[0],:] = clean_features[range(np.shape(noisy_features)[0]),:]
        iter = iter + np.shape(noisy_features)[0]

    # Normalize input output data 
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()
    X_train_norm = input_scaler.fit_transform(X_train[range(iter),:])
    y_train_norm = output_scaler.fit_transform(y_train[range(iter),:])

    pickle.dump(input_scaler, open(output_dir + '/input_scaler.dat', 'wb'))
    pickle.dump(output_scaler, open(output_dir + '/output_scaler.dat', 'wb'))

    # Train the model
    # On définit l'architecture du réseau de neurones
    regr = MLPRegressor(
                        hidden_layer_sizes=mlp_arch,        #architecture du MLP
                        activation=activation_function,     #fct d'activation               
                        solver=solver,                      #optimisateur
                        max_iter=max_iter,                  #nb d'iteration max
                        random_state=seed,                  #seed, init aléatoire
                        alpha = alpha                       #regularisation
                       )                
     
    # on entraine le modele avec la fonction fit
    regr.fit(X_train_norm, y_train_norm)

    pickle.dump(regr, open(output_dir + '/regr.dat', 'wb'))

    #on verifie ici le training 
    print(f'Final training loss: {regr.loss_}')
    

################################################
def do_test(ind):
    if isdir(output_dir + '/flac_synth') is False:
        mkdir(output_dir + '/flac_synth')
        
    regr = pickle.load(open(output_dir + '/regr.dat', 'rb'))

    
    input_scaler = pickle.load(open(output_dir + '/input_scaler.dat', 'rb'))
    output_scaler = pickle.load(open(output_dir + '/output_scaler.dat', 'rb'))

    all_noisy_features_filenames = sorted(listdir_fullpath(output_dir + '/noisy_features'))
    all_clean_features_filenames = sorted(listdir_fullpath(output_dir + '/clean_features'))
    
    all_clean_features_predicted = []
    all_clean_features_original = []
    
    for f in ind:
        noisy_features = np.load(all_noisy_features_filenames[f])
        clean_features = np.load(all_clean_features_filenames[f])
        clean_features_predicted = output_scaler.inverse_transform(regr.predict(input_scaler.transform(noisy_features)))

        #Stocker l'ensemble des séquences cibles prédites (clean_features_predicted) ainsi que les séquences cibles originales (dans 2 numpy array distinctes).
        all_clean_features_predicted.append(clean_features_predicted)
        all_clean_features_original.append(clean_features)
        
        if step_synth:
            y_noisy = librosa.feature.inverse.mel_to_audio(noisy_features.transpose(),sr=audio_fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, power=2.0, n_iter=64,fmin=fmin, fmax=fmax)
            sf.write(output_dir + '/flac_synth/' + splitext(basename(all_noisy_features_filenames[f]))[0] + "_noisy_anasyn.flac", y_noisy, audio_fs, 'PCM_16')

            y_clean = librosa.feature.inverse.mel_to_audio(clean_features.transpose(),sr=audio_fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, power=2.0, n_iter=64,fmin=fmin, fmax=fmax)
            sf.write(output_dir + '/flac_synth/' + splitext(basename(all_clean_features_filenames[f]))[0] + "_clean_anasyn.flac", y_clean, audio_fs, 'PCM_16')

            y_clean_predicted = librosa.feature.inverse.mel_to_audio(clean_features_predicted.transpose(),sr=audio_fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, power=2.0, n_iter=32,fmin=fmin, fmax=fmax)
            sf.write(output_dir + '/flac_synth/' + splitext(basename(all_clean_features_filenames[f]))[0] + "_synth.flac", y_clean_predicted, audio_fs, 'PCM_16')

        #TODO: Evaluer la performance globale du système de conversion en calculant l'erreur quadratique moyenne entre les mel-spectrogrammes cibles prédits et les mel-spectrogramme originaux (par exemple à l'aide de la fonction sklearn.metrics.mean_squared_error). 
    
    # Convertir les listes en numpy arrays pour le calcul global
    all_clean_features_predicted = np.vstack(all_clean_features_predicted)
    all_clean_features_original = np.vstack(all_clean_features_original)
    # Calculer l'erreur quadratique moyenne (MSE) entre les spectrogrammes cibles originaux et prédits
    mse = mean_squared_error(all_clean_features_original, all_clean_features_predicted)
    print("Erreur quadratique moyenne (MSE) entre les spectrogrammes cibles et prédits :", mse)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(clean_features.T, sr=audio_fs, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Melspectrogramme original sans bruit')
    plt.show()
    plt.figure(figsize=(10, 4))

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(clean_features_predicted.T, sr=audio_fs, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Melspectrogramme prédit et débruité')
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(y_clean_predicted)  # Assurez-vous que y_clean_predicted est la forme d'onde reconstruite
    plt.title('Forme d\'onde associée au MelSpectrogramme débruité')
    plt.xlabel('Temps')
    plt.ylabel('Amplitude')
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(y_clean)  # Assurez-vous que y_clean_synth est la forme d'onde reconstruite
    plt.title('Forme d\'onde associée au MelSpectrogramme original sans bruit ')
    plt.xlabel('Temps')
    plt.ylabel('Amplitude')
    plt.show()

######################################################################################
# Main
if __name__ == '__main__':
    if isdir(output_dir) is False:
        mkdir(output_dir)
    
    if step_extract_features: 
        print('Extracting audio features (noisy speaker)')
        extract_features(noisy_audio_dir,output_dir + "/noisy_features/")

        print('Extracting audio features (clean speaker)')
        extract_features(clean_audio_dir,output_dir + "/clean_features/")

    if step_train:
        train_ind, test_ind = train_test_split(range(nb_sentences),test_size=0.10)
        np.save(output_dir + '/train_ind.npy',train_ind)
        np.save(output_dir + '/test_ind.npy',test_ind)
        print(f'Train MLP regressor: ({mlp_arch})')
        do_train(train_ind)
        

    if step_test:
        print('MLP-based regression and audio synthesis')
        test_ind = np.load(output_dir + '/test_ind.npy')
        do_test(test_ind)

        




















# clean_tensor = process_flac_files(path_clean,n_fft)
# noisy_tensor = process_flac_files(path_noisy,n_fft)
# noisy_test_tensor = process_flac_files(path_noisy_test,n_fft)
# clean_test_tensor = process_flac_files(path_clean_test,n_fft)

# if do_train:
#     #initialisation du modèle
#     mlp = MLPRegressor(
#         hidden_layer_sizes=(64,),  # Taille de la couche cachée (ici, une seule couche de 128 neurones)
#         activation='relu',          # Fonction d'activation
#         solver='adam',              # Optimiseur
#         max_iter=200,                # Nombre maximum d'itérations
#         random_state=42
#     )

#     #entrainement
#     mlp.fit(noisy_tensor,clean_tensor)
#     #on enregistre le modèle
#     pickle.dump(mlp,open('modele/mlp.dat','wb'))


# #test
# mlp = pickle.load(open('modele/mlp.dat', 'rb'))
# predicted_tensor = mlp.predict(noisy_test_tensor)
# mse = mean_squared_error(clean_test_tensor, predicted_tensor)
# print(mse)

# #mise en forme des résulats

# predicted_tensor = predicted_tensor.T
# # Spécifier un répertoire de sauvegarde
# predicted_spect_folder = "Result/predicted_spect/"
# # Vérifier si le répertoire existe, sinon le créer
# if not os.path.exists(predicted_spect_folder):
#     os.makedirs(predicted_spect_folder)

# #récupération des noms originaux: 
# noisy_test_filenames = os.listdir(path_noisy_test)
# segment_height = predicted_tensor.shape[0]
# segment_width = int(predicted_tensor.shape[1]/len(noisy_test_filenames))
# nombre_segment = int(predicted_tensor.shape[1]/segment_width)
# for i in range(nombre_segment):
#     segment = predicted_tensor[:,i * segment_width:(i + 1) * segment_width]
#     filename = noisy_test_filenames[i]
#     filename = filename.replace(".flac","")
    
#     # Sauvegarder chaque segment dans le répertoire spécifié
#     np.save(os.path.join(predicted_spect_folder, filename), segment)


# # Spécifier un répertoire de sauvegarde
# predicted_flac_folder = "Result/predicted_flac/"
# # Vérifier si le répertoire existe, sinon le créer
# if not os.path.exists(predicted_flac_folder):
#     os.makedirs(predicted_flac_folder)

# spect_to_flac(predicted_spect_folder,predicted_flac_folder)

