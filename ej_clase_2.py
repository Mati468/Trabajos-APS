#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:55:33 2023

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la simulación

fs = 250 # Hz
N = fs

delta_f = fs / N # resolucion espectral en [Hz]

Ts = 1/fs
T_simulacion = N * Ts # [s]


# Parámetros de la señal
fx = 1 # Hz
ax = np.sqrt(2)# V


# grilla temporal
tt = np.arange(start = 0, stop = T_simulacion, step = Ts)

xx = ax * np.sin( 2 * np.pi * fx * tt )

#plt.plot(tt, xx)


x_2 = np.mean(xx) #media/promedio
x_3 = np.std(xx) #desvio estandar
x_5 = xx/x_3 # para cualquier señal, primero estandarizo el vector de mi grafica,
x_4= np.var(x_5) # luego obtengo la varianza, que 

print("Promedio: ", x_2)
print("Desvio: ", x_3)
print("Varianza: ", x_4)

x_3
