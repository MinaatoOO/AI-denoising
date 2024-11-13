"""
Nom du Fichier : test_plot.py
Description    : à pour but de tester les afficheurs d'un fichier .FLAC
Auteur         : Bounsavath
Date           : 13 11 2024
Version        : 1.0
"""
from project_tools.plot import *

path = "DATA/DataTotale/0.flac"
plot_waveform(path)
plot_spect(path)
plot_melspect(path)