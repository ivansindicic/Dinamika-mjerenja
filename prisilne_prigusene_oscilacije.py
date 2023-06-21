# prisilne_prigusene_oscilacije.py
### Analiza troetazne zgrade s razlicitim masama stropova

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from frekvencija import m,k,omega
from slobodne_prigusene_oscilacije import zeta_average

## Troetazni problem (labos)
# Ucitavanje podataka mjerenja iz labosa

# Mijenjanje ispitivanja (2 ili 4)
file = "mjerenje2_pomak_x.csv" #lokalna datoteka u radnom dokumentu

t_exp = []
xb_exp = []
x1_exp = []
x2_exp = []
x3_exp = []

# Odabir podataka ovisno o ispitivanju
if file == "mjerenje2_pomak_x.csv":
    with open(file) as f: #sintaksa za otvaranje datoteka
        for line in f:
            row = line.split(",")
            t_exp.append(float(row[0]))
            xb_exp.append((float(row[1])-7)/1000)
            x1_exp.append((float(row[2])-float(row[1]))/1000)
            x2_exp.append((float(row[3])-float(row[1]))/1000)
            x3_exp.append((float(row[4])-float(row[1]))/1000)
    row_start = 484 #red pocetka ispitivanja 2
    row_end = 3235 #red kraja ispitivanja 2
elif file == "mjerenje4_pomak_x.csv":
    with open(file) as f: #sintaksa za otvaranje datoteka
        for line in f:
            row = line.split(",")
            t_exp.append(float(row[0]))
            xb_exp.append((float(row[1]))/1000)
            x1_exp.append((float(row[2])-float(row[1]))/1000)
            x2_exp.append((float(row[3])-float(row[1]))/1000)
            x3_exp.append((float(row[4])-float(row[1]))/1000)
    row_start = 468 #red pocetka ispitivanja 4
    row_end = 3243 #red kraja ispitivanja 4

t_start = t_exp[row_start] #pocetak mjerenja ispitivanja
t_end = t_exp[row_end]-t_exp[row_start] #kraj mjerenja ispitivanja
dt = 0.01 #veličina vremenskog koraka
N = int(t_end/dt) #broj vremenskih koraka

t_new = []
for i in range (len(t_exp)):
    t_new.append(t_exp[i]-t_start)
    
xb_new = xb_exp[row_start:row_end]
x1_new = x1_exp[row_start:row_end]
x2_new = x2_exp[row_start:row_end]
x3_new = x3_exp[row_start:row_end]

## Troetazni problem (izracun)

# Prigusenje prisilnih oscilacija
omega_i = omega[0]
omega_j = omega[1]
a0 = zeta_average*(2*omega_i*omega_j)/(omega_i+omega_j)
a1 = zeta_average*(2/(omega_i + omega_j))
c = np.dot(m,a0) + np.dot(k,a1) #matrica prigusenja

e = np.array([[1], 
              [1], 
              [1]])

# Newmarkovi parametri:
beta=1/4
gama=1/2
b1 = 1/(beta*dt**2)
b2 = -1/(beta*dt)
b3 = 1-(1/(2*beta))
b4 = gama/(beta*dt)
b5 = 1-(gama/beta)
b6 = dt*(1-(gama/(2*beta)))

x0 = [x1_exp[0], x2_exp[0], x3_exp[0]] #pomak
v0 = [0, 0, 0] #brzina
a0 = np.dot(inv(m), np.dot(k, x0)) #ubrzanje

# Prazni matrice u koje cemo spremati t, x, v i a u svakom koraku:
t = np.zeros(N)
x = np.zeros([3,N])
v = np.zeros((3,N))
a = np.zeros((3,N))

# Pocetni uvjeti
x[:,0] = x0
v[:,0] = v0
a[:,0] = a0

# Efektivna matrica krutosti (konstantna je):
keff = (m*b1 + k + c*b4)
keff_inv = inv(keff)

vb_new = [(xb_new[i + 1] - xb_new[i])/dt for i in range(len(xb_new)-1)]
ab_new = [(vb_new[i + 1] - vb_new[i])/dt for i in range(len(vb_new)-1)]

# Petlja kojom koracamo u vremenu i racunamo nepoznate pomake, brzine i ubrzanja u svakom trenutku:
for i in range (1,N):
    F = -ab_new[i-2]*np.dot(m,e)
    feff = np.transpose(F) + (np.dot(m,(b1*x[:,i-1] - b2*v[:,i-1] - b3*a[:,i-1]))) + (np.dot(c,(b4*x[:,i-1] - b5*v[:,i-1] - b6*a[:,i-1])))
    xt = np.dot(keff_inv,feff[0,:])
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
# Izgled grafa
plt.xlim([0,t_end])
plt.title('Prisilne prigušene oscilacije')
plt.xlabel('Vrijeme [s]')
plt.ylabel('Pomak x [m]')
plt.axhline(0, color='red', linestyle='--')
plt.legend(loc ="lower right")
plt.grid()
plt.show()