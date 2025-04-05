# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 21:29:01 2025

@author: Nancy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:58:13 2025

@author: mariano
"""

#%% módulos y funciones a importar
import numpy as np
import matplotlib.pyplot as plt


#%% Datos de la simulación

fs = 1000  # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras

# Datos del ADC
B = 8 # bits
Vf=2
#Vf = np.sqrt(12)*2**7 # rango simétrico de +/- Vf Volts
#q = np.sqrt(12) # paso de cuantización de q Volts
q=Vf/(2**7)
x = np.var
# datos del ruido (potencia de la señal normalizada, es decir 1 W)
pot_ruido_cuant = 1# Watts 
kn = 1. # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn #Una es potencia de otra 

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral


#%% Experimento: 
"""
   Se desea simular el efecto de la cuantización sobre una señal senoidal de 
   frecuencia 1 Hz. La señal "analógica" podría tener añadida una cantidad de 
   ruido gausiano e incorrelado.
   
   Se pide analizar el efecto del muestreo y cuantización sobre la señal 
   analógica. Para ello se proponen una serie de gráficas que tendrá que ayudar
   a construir para luego analizar los resultados.
   
"""

# np.random.normal
# np.random.uniform


# Señales
# grilla temporal
T_simulacion = N * ts # [s]
tt = np.arange(start = 0, stop = T_simulacion, step = ts)
fx=1.5
ax = np.sqrt(2)
analog_sig = ax * np.sin( 2 * np.pi * fx * tt )# señal analógica sin ruido
na = np.random.normal(0, 1, size=1000) # ruido analoogico
sr = analog_sig + na# señal analógica de entrada al ADC (con ruido analógico) (el profe lallama na) el ruido es aleatorio entonces este va a ser numeros aleatorios con media y varianza 
'''
aca hay que pensar como cuantizar. la idea es escalar mi seal en 256 valores.
de -128 a 128. 


'''




srq = np.round(sr/q)*q  # señal cuantizada




nn =  na# señal de ruido de analógico
nq = srq - sr # señal de ruido de cuantización. 

x_2 = np.mean(na) #media/promedio
x_3 = np.std(na) #desvio estandar
#x_5 = xx/x_3 # para cualquier señal, primero estandarizo el vector de mi grafica,
x_4= np.var(na) # luego obtengo la varianza, que 

print("Promedio: ", x_2)
print("Desvio: ", x_3)
print("Varianza: ", x_4)

#%% Visualización de resultados

# ###########
# # Espectro
# ###########

plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr) #escalamiento de antitransformada innecesario, 
#con cantidad igual a muestras y valores complejos
ft_Srq = 1/N*np.fft.fft( srq) 
ft_As = 1/N*np.fft.fft( analog_sig)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
# plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
# plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
# plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

