# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 19:50:04 2025

@author: Nancy
"""

import numpy as np
import matplotlib.pyplot as plt

# Datos generales de la simulación
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras
f0=2001

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

# grilla de sampleo temporal
tt = np.linspace(0, (N-1)*ts, N)

argg=2*np.pi*f0*tt
xx=np.sin(argg)

plt.plot(tt,xx)
