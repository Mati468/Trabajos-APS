# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:33:26 2025

@author: Nancy
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# Parámetros de simulacion
N = 1000  # Número de muestras por realización
pruebas = 200   # Número de realizaciones
SNR = 10  # en dB

# Parámetros de las señales
a1 = sqrt(2)  # Amplitud de la señal
Ω0 = np.pi / 2
P_ruido=1/SNR
k=np.zeros(pruebas,N)# Índide temporal
Fr=np.random.uniform(-1/2,1/2,pruebas)
t=np.arange(0, N, 1)# 0 a N-1 con paso de 1
Ω1= Ω0 + Fr*(2*np.pi/N)


