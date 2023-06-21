# slobodne_neprigusene_oscilacije.py
### Analiza troetazne zgrade s razlicitim masama stropova

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from frekvencija import m,k
from frekvencija import ton1,ton2,ton3
from slobodne_prigusene_oscilacije import t_end,dt,N
from slobodne_prigusene_oscilacije import x1_new,x2_new,x3_new

## Troetazni problem (mjerenja)

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
x[:,0] = x0
v[:,0] = v0
a[:,0] = a0

# Efektivna matrica krutosti (konstantna je):
keff = (m*b1 + k)
keff_inv = inv(keff)

# Petlja kojom koracamo u vremenu i racunamo nepoznate pomake, brzine i ubrzanja u svakom trenutku:
for i in range (1,N):
    feff = (np.dot(m,(b1*x[:,i-1] - b2*v[:,i-1] - b3*a[:,i-1])))
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
# Izgled grafa
plt.xlim([0,t_end])
plt.title('Slobodne neprigušene oscilacije')
plt.xlabel('Vrijeme [s]')
plt.ylabel('Pomak x [m]')
plt.axhline(0, color='red', linestyle='--')
plt.legend(loc ="lower right")
plt.grid()
plt.show()