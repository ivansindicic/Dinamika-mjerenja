# slobodne_prigusene_oscilacije.py
### Analiza troetazne zgrade s razlicitim masama stropova

import math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from frekvencija import m,k,omega
from frekvencija import ton1,ton2,ton3

## Troetazni problem (mjerenja)
# Ucitavanje podataka mjerenja iz laboratorija

# Mijenjanje ispitivanja (1 ili 3)
file = "mjerenje1_pomak_x.csv" #lokalna datoteka u radnom dokumentu

t_exp = []
xb_exp = []
x1_exp = []
x2_exp = []
x3_exp = []

with open(file) as f: #sintaksa za otvaranje datoteka
    for line in f:
        row = line.split(",") #razdvajanje teksta temeljem razmaka ili taba
        t_exp.append(float(row[0]))
        xb_exp.append((float(row[1])-float(row[1]))/1000)
        x1_exp.append((float(row[2])-float(row[1]))/1000)
        x2_exp.append((float(row[3])-float(row[1]))/1000)
        x3_exp.append((float(row[4])-float(row[1]))/1000)

# Granice i intervali vremena ispitivanja
t_start = t_exp[x3_exp.index(max(x3_exp))] #pocetak mjerenja ispitivanja
t_end = t_exp[-1] - t_exp[1] #kraj mjerenja ispitivanja
dt = 0.01 #veličina vremenskog koraka
N = int(t_end/dt) #broj vremenskih koraka

t_new = []
for i in range (len(t_exp)):
    t_new.append(t_exp[i]-t_start)

x1_new = x1_exp[x3_exp.index(max(x3_exp)):]
x2_new = x2_exp[x3_exp.index(max(x3_exp)):]
x3_new = x3_exp[x3_exp.index(max(x3_exp)):]

## Troetazni problem (izracun)

# Prigusenje slobodnih oscilacija
x3_exp_values = [value for value in x3_new if value > 0]
x = np.arange(0, t_end, dt)
x3_new_array = np.array(x3_new)
max_indices = np.argwhere(np.diff(np.sign(np.diff(x3_new))) < 0).flatten()
x_max = x[max_indices]
y_max = x3_new_array[max_indices]
y_max_values = y_max.tolist()

zeta_list = []

for i in range(len(y_max_values) - 1):
    period = y_max_values[i] / y_max_values[i + 1]
    zeta = (1/(2*math.pi))*math.log(period)
    zeta_list.append(zeta)

zeta_average = sum(zeta_list)/len(zeta_list)
omega_d = omega*(1-zeta_average**2)
omega_i = omega_d[0]
omega_j = omega_d[1]
a0 = zeta_average*(2*omega_i*omega_j)/(omega_i+omega_j)
a1 = zeta_average*(2/(omega_i + omega_j))
c = np.dot(m,a0) + np.dot(k,a1) #matrica prigusenja

# Prosječno trajanje jednog intervala
interval = np.diff(max_indices)
T = np.mean(interval)

# Newmarkovi parametri:
beta=1/4
gama=1/2
b1=1/(beta*dt**2)
b2=-1/(beta*dt)
b3=1-(1/(2*beta))
b4=gama/(beta*dt)
b5=1-(gama/beta)
b6=dt*(1-(gama/(2*beta)))

x0 = [x1_new[0], x2_new[0], x3_new[0]] # pomak
v0 = [0, 0, 0] # brzina
a0 = np.dot(inv(m), np.dot(k, x0)) # ubrzanje

# Prazni matrice u koje cemo spremati t, x, v i a u svakom koraku:
t = np.zeros(N)
x = np.zeros([3,N])
v = np.zeros((3,N))
a = np.zeros((3,N))

# Pocetni uvjeti
x[:,0] = x0 #x0 za mjerenja; ton1,ton2,ton3 za oblik osciliranja
v[:,0] = v0
a[:,0] = a0

# Efektivna matrica krutosti (konstantna je):
keff = (m*b1 + k + c*b4)
keff_inv = inv(keff)

# Petlja kojom koracamo u vremenu i racunamo nepoznate pomake, brzine i ubrzanja u svakom trenutku:
for i in range (1,N):
    feff = (np.dot(m,(b1*x[:,i-1] - b2*v[:,i-1] - b3*a[:,i-1]))) + (np.dot(c,(b4*x[:,i-1] - b5*v[:,i-1] - b6*a[:,i-1])))
    xt = np.dot(keff_inv,feff)
    x[:, i] = xt[:]
    vt = np.dot(b4,(x[:,i] - x[:,i-1])) + np.dot(b5,v[:,i-1]) + np.dot(b6,a[:,i-1])
    v[:,i]=vt[:]
    at = np.dot(b1,(x[:,i]-x[:,i-1])) + np.dot(b2,v[:,i-1]) + np.dot(b3,a[:,i-1])
    a[:,i]=at
    t[i]=i*dt

# CRTANJE GRAFOVA
# Podaci troetaznog problema (izracun)
plt.plot (t, x[0,:], label='Kat1(izracun)')
plt.plot (t, x[1,:], label='Kat2(izracun)')
plt.plot (t, x[2,:], label='Kat3(izracun)')
# Podaci troetaznog problema (mjerenja)
plt.plot(t_new,xb_exp, label='Baza(mjerenja)')
plt.plot(t_new, x1_exp, label='Kat1(mjerenja)')
plt.plot(t_new, x2_exp, label='Kat2(mjerenja)')
plt.plot(t_new, x3_exp, label='Kat3(mjerenja)')
# plt.scatter(x_max, y_max, color='red', label='Maximum')
# Izgled grafa
plt.xlim([0,t_end])
plt.title('Slobodne prigušene oscilacije')
plt.xlabel('Vrijeme [s]')
plt.ylabel('Pomak x [m]')
plt.axhline(0, color='red', linestyle='--')
plt.legend(loc ="lower right")
plt.grid()
plt.show()