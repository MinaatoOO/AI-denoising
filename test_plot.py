"""
Nom du Fichier : test_plot.py
Description    : à pour but de tester les afficheurs d'un fichier .FLAC
Auteur         : Bounsavath
Date           : 13 11 2024
Version        : 1.0
"""
from project_tools.plot import *
import matplotlib.pyplot as plt
import os
import torchaudio
import torch
import numpy as np
from tqdm import tqdm  # Importer tqdm pour la barre de progression
import random

path = "DATA/train/noisy/10.flac"
plot_waveform(path)
plot_spect(path)
plot_melspect(path)
