# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:32:16 2025

@author: Matías Cassino
"""
#%% CONSIGNA
"""
En este primer trabajo comenzaremos por diseñar un generador de señales que utilizaremos en las primeras simulaciones que hagamos.
La primer tarea consistirá en programar una función que genere señales senoidales y que permita parametrizar:
vmax = la amplitud máxima de la senoidal (volts)
dc = su valor medio (volts)
ff = la frecuencia (Hz)
ph = la fase (radianes)
nn = la cantidad de muestras digitalizada por el ADC (# muestras)
fs = la frecuencia de muestreo del ADC.
es decir que la función que uds armen debería admitir se llamada de la siguiente manera

tt, xx = mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs)
Recuerden que tanto xx como tt deben ser vectores de Nx1. Puede resultarte útil el módulo de visualización
matplotlib.pyplot donde encontrarán todas las funciones de visualización estilo Matlab. Para usarlo:

import matplotlib.pyplot as plt
plt.plot(tt, xx)

Realizar los experimentos que se comentaron en clase. Siguiendo la notación de la función definida más arriba:
ff = 500 Hz
ff = 999 Hz
ff = 1001 Hz
ff = 2001 Hz
Implementar alguna otra señal propia de un generador de señales. 
"""
#%% LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#%% FUNCIÓN SEÑAL SENOIDAL
def mi_funcion_sen( vmax, dc, ff, ph, nn, fs):
    # Datos generales de la simulación
    ts = 1/fs # tiempo de muestreo
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (nn-1)*ts, nn).reshape(nn, 1)
    
    # matriz asociada a la señal senoidal
    argg = 2*np.pi*ff*tt + ph
    xx = vmax*(np.sin(argg) + dc).reshape(nn, 1)
    
    return tt,xx

#%% FUNCION SEÑAL CUADRADA
def mi_funcion_square( vmax, dc, ff, ph, nn, fs):
    # Datos generales de la simulación
    ts = 1/fs # tiempo de muestreo
    
    # grilla de sampleo temporal
    t_square = np.linspace(0, (nn-1)*ts, nn).reshape(nn, 1)
    
    # matriz asociada a la señal senoidal
    argg = 2*np.pi*ff*tt + ph
    x_square = vmax*(signal.square(argg) + dc).reshape(nn, 1)
    
    return t_square,x_square

    
#%% PARAMETROS (interpreté que la consigna pide mantener los 4 primeros parámetros constantes mientras que los dos restantes son configurables)
N = 1000
fs = 1000.0

#%% INVOCO LA FUNCION DE LA SEÑAL SENOIDAL
tt, xx = mi_funcion_sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = N, fs = fs)

#%% INVOCO LA FUNCION DE LA SEÑAL CUADRADA
t_square, x_square = mi_funcion_square(vmax = 2, dc = 0, ff = 5, ph = np.pi/4, nn = N, fs = fs)

#%% PRESENTACION DE RESULTADOS SEÑAL SENOIDAL
plt.figure(1)
plt.plot(tt,xx,label = 'ff = 1 Hz')
plt.grid()
plt.legend()
plt.title('Señal senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')

#%% REPITO EXPERIMENTO SIN USO DE FOR O WHILE
tt_500, xx_500 = mi_funcion_sen(vmax = 1, dc = 0, ff = 500, ph = 0, nn = N, fs = fs)
tt_999, xx_999 = mi_funcion_sen(vmax = 1, dc = 0, ff = 900, ph = 0, nn = N, fs = fs)
tt_1001, xx_1001 = mi_funcion_sen(vmax = 1, dc = 0, ff = 1001, ph = 0, nn = N, fs = fs)
tt_2001, xx_2001 = mi_funcion_sen(vmax = 1, dc = 0, ff = 2001, ph = 0, nn = N, fs = fs)

#%% PRESENTACION DE RESULTADOS PARA DISTINTAS FRECUENCIAS SIN FOR NI WHILE
plt.figure(2)
plt.plot(tt_500,xx_500,label = 'ff = 500 Hz',color='magenta')
plt.grid()
plt.legend()
plt.title('Señales senoidales para distintas frecuencias')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.figure(3)
plt.plot(tt_999,xx_999,label = 'ff = 999 Hz',color='red')
plt.grid()
plt.legend()
plt.title('Señales senoidales para distintas frecuencias')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.figure(4)
plt.plot(tt_1001,xx_1001,label = 'ff = 1001 Hz',color='green')
plt.grid()
plt.legend()
plt.title('Señales senoidales para distintas frecuencias')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.figure(5)
plt.plot(tt_2001,xx_2001,label = 'ff = 2001 Hz',color='orange')
plt.grid()
plt.legend()
plt.title('Señales senoidales para distintas frecuencias')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')

#%% PRESENTACION DE RESULTADOS SEÑAL CUADRADA
plt.figure(6)
plt.plot(t_square,x_square,label = 'ff = 1000 Hz',color='yellow')
plt.grid()
plt.legend()
plt.title('Señal cuadrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')