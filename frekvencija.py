# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 18:18:57 2023

@author: petrovic, pranjic, sindicic
"""

### Analiza troetazne zgrade s razlicitim masama stropova

import math
import numpy as np
import scipy.linalg as sc
import matplotlib.pyplot as plt

## Troetazni problem (izracun)

# Ocitane vrijednosti
sila = [0.04, 1.25, 3.03, 4.20, 5.61, 6.92, 7.87, 9.34, 10.92] # x-axis vrijednosti
pomak = [5.3, 11.3, 13.4, 15.3, 17.6, 17.9, 18.7, 19.6, 22.5] # y-axis vrijednosti

# Izracun koeficijenata polinoma
coefficients = np.polyfit(sila, pomak, 1)

# Funkcija koja se može koristiti za predviđanje y-vrijednosti linije trenda
trendline_function = np.poly1d(coefficients)

# # CRTANJE GRAFOVA
# fig, ax = plt.subplots()
# plt.scatter(sila, pomak)
# ax.plot(sila, pomak, label='Vrijednosti ispitivanja')
# ax.set_xlabel('Sila [N]')
# ax.set_ylabel('Pomak [cm]')
# ax.set_title('Sila - Pomak')
# ax.legend(loc ="lower right")
# plt.plot(sila, trendline_function(sila), color='red')
# plt.text(0.5, 5, f'y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}', fontsize=12, color='red')
# plt.show()

# Parametri vijka
mv = 2.3 #masa (g)
# Parametri pleksiglas ploce
mpp = 140.6 #masa (g)
depp = 0.14 #debljina (cm)
dupp = 30.45 #duzina (cm)
# Parametri aluminijske ploce
mpa = 368.1 #masa (g)
deap = 1 #debljina (cm)
duap = 360.639 #duzina (cm)
# Dodana masa (g)
# Dodana masa = 200 g
dm = 0

# Krutost
# *1000 mm->m
krutost = coefficients[0]*1000

# Izracun masa
# 10 vijaka po katu gdje su dvije pleksiglas ploce (mv)
# 4 vijaka po katu gdje je jedna pleksiglas ploca (mv)
# 2 pleksiglas ploce na prva dva kata (mpp)
# 1 pleksiglas ploca na zadnjem katu (mpp)
# 2 aluminijske ploce po katu (mpa)
# 1 aluminijska ploca na zadnjem katu (mpa)
# mnoziti sa 0.001 g->kg
masa1 = (10*mv + 2*mpp + 2*mpa)*0.001
masa2 = (10*mv + 2*mpp + 2*mpa)*0.001
masa3 = (4*mv + 1*mpp + 1*mpa + dm)*0.001

# Matrica masa
m = np.array([[masa1, 0, 0], 
              [0, masa2, 0], 
              [0, 0, masa3]])

# Matrica krutosti
k = krutost * np.array([[2, -1, 0], 
                        [-1, 2, -1], 
                        [0, -1, 1]])

w, v = sc.eigh(k, m)

ton1 = v[:,0]
ton2 = v[:,1]
ton3 = v[:,2]

# Prirodne frekvencije
omega = np.sqrt(w)

# Frekvencije
f = omega/(2*math.pi)

# Periodi
T = 1/f

# print('Krutost: ', f"{krutost:.1f}", 'N/m')
# print('Matrica krutosti:')
# print(w)
# print("k1: ", f"{w[0]:.2f}", 'N/m')
# print("k2: ", f"{w[1]:.2f}", 'N/m')
# print("k3: ", f"{w[2]:.2f}", 'N/m')
# print('Mase: ')
# print('m1: ', f"{masa1:.4f}",'kg')
# print('m2: ', f"{masa2:.4f}",'kg')
# print('m3: ', f"{masa3:.4f}",'kg')
# print('Matrica masa: ')
# print(v)
# print("Kružna frekvencija: ")
# print("w1: ", f"{omega[0]:.2f}", 'rad/s')
# print("w2: ", f"{omega[1]:.2f}", 'rad/s')
# print("w3: ", f"{omega[2]:.2f}", 'rad/s')
# print("Periodi: ")
# print("T1: ", f"{T[0]:.2f}", 's')
# print("T2: ", f"{T[1]:.2f}", 's')
# print("T3: ", f"{T[2]:.2f}", 's')
# print("Frekvencija: ")
# print("f1: ", f"{f[0]:.2f}", 'Hz')
# print("f2: ", f"{f[1]:.2f}", 'Hz')
# print("f3: ", f"{f[2]:.2f}", 'Hz')