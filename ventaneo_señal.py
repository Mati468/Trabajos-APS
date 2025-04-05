# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:20:48 2025

@author: Nancy
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

# Señales
# grilla temporal
T_simulacion = N * ts # [s]
tt = np.arange(start = 0, stop = T_simulacion, step = ts)
fx=259.5
ax = np.sqrt(2)
analog_sig = ax * np.sin( 2 * np.pi * fx * tt )# señal analógica sin ruido
na = np.random.normal(0, 1, size=1000) # ruido analoogico
sr = analog_sig + na# señal analógica de entrada al ADC (con ruido analógico) (el profe lallama na) el ruido es aleatorio entonces este va a ser numeros aleatorios con media y varianza 

x_3 = np.std(analog_sig) #desvio estandar
print("Desvio: ", x_3)

#%% Visualización de resultados

# ###########
# # Espectro
# ###########

plt.figure(2)
ft_As = 1/N*np.fft.fft( analog_sig)
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), ls='dotted', label='$ s $ (sig.)' )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()


# grilla de sampleo frecuencial
ff1 = np.linspace(0, (N-1)*df, N)

na_w=analog_sig*np.bartlett (N) #señal enventanada

x_5 = na_w/x_3 # normalizo señal enventanada

fft_senal=(1/N)*np.fft.fft(x_5)

plt.plot(ff1[bfrec], 10 * np.log10(2 * np.abs(fft_senal[bfrec])**2), ls='dotted', label='$ s $ (sig. con Bartlett)')
plt.legend()