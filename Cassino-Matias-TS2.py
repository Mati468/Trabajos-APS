# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:43:56 2025

@author: Matías Cassino
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, bode

#%% FUNCIONES TRANSFERENCIA CONSIDERANDO L=R=C=1
numerador_a=[0, 1, 0]
denominador_a=[1, 1, 1]

numerador_b=[1, 0, 0]
denominador_b=[1, 1, 1]

H_a=TransferFunction(numerador_a, denominador_a)
H_b=TransferFunction(numerador_b, denominador_b)

#%%DETERMINACIÓN DE RESPUESTA EN MÓDULO Y FASE
w_a, modulo_a, fase_a = bode(H_a)
w_b, modulo_b, fase_b = bode(H_b)

#%%CONVIERTO FASES EN GRADOS A RADIANES
fase_a_rad=(fase_a*np.pi)/180
fase_b_rad=(fase_b*np.pi)/180

#%%PENDIENTE DE ASÍNTOTAS
asintota_a=-20*np.log10(w_a)
asintota_b=np.zeros_like(w_b) #array de ceros con tamaño de w_b


#%%GRÁFICOS DE MÓDULOS
plt.figure(1)
plt.semilogx(w_a, modulo_a, label='H_a(s) = 1 / (s² + s + 1)', color='red')
plt.semilogx(w_a, asintota_a, label='Asintota de H_a(jw)', color='blue', linestyle='dashed')
plt.xlabel('w [rad/s]')
plt.ylabel('20log|H| [dB]')
plt.title('Diagrama de Bode (Módulo H_a(s))')
plt.grid()
plt.legend()  

plt.figure(2)
plt.semilogx(w_b, modulo_b, label='H_b(jw) = s² / (s² + s + 1)', color='blue')
plt.semilogx(w_b, asintota_b, label='Asintota de H_b(jw)', color='red', linestyle='dashed')
plt.xlabel('w [rad/s]')
plt.ylabel('20log|H| [dB]')
plt.title('Diagrama de Bode (Módulo H_b(jw))')
plt.grid()
plt.legend()  

#%%GRÁFICO DE FASES
plt.figure(3)
plt.semilogx(w_a, fase_a_rad, label='H_a(s) = 1 / (s² + s + 1)', color='red')  
plt.xlabel('w [rad/s]')
plt.ylabel('Fase [rad]')  
plt.title('Diagrama de Bode (Fase H_a(jw))')  
plt.grid()  
plt.legend()

plt.figure(4)
plt.semilogx(w_b, fase_b_rad, label='H_b(s) = s² / (s² + s + 1)', color='blue') 
plt.xlabel('w [rad/s]')
plt.ylabel('Fase [rad]')  
plt.title('Diagrama de Bode (Fase H_b(jw))')  
plt.grid()  
plt.legend()