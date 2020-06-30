#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pylab, math
import pandas as pd
import numpy as np
from fitter import Fitter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#Funciones de prueba
#Gaussiana
def gaussiana(X, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((X-mu)/sigma)**2)

#Rayleigh
def rayleigh(X, a, b):
    return (2/b)*(X-a)*np.exp(-((X-a)**2)/b)

#Cuadratica
def cua(X, a, b, c):
    return a*X**2 +b*X+c

#Nombre de los archivos que contienen los datos. En caso de ser necesario, agregar la dirección
archivo1='xy.csv'
archivo2='xyp.csv'

#Abrimos los archivos con los datos
datosxy = pd.read_csv (archivo1)
datospxy = pd.read_csv (archivo2)

#Se crea un vector x, un vector y, las probabilidades independientes px, py, y
#la probabilidad combinada pxy
px=[]
x=[]
py=[]
y=[]
pxy=[]
datospxy.set_index('x', inplace=True)

for i in range(5,16):
    dato=datospxy.loc[i]
    px.append(dato['p'].sum())
    x.append(i)
    pxy.append([])

datospxy.set_index('y', inplace=True)
for i in range(5,26):
    dato=datospxy.loc[i]
    py.append(dato['p'].sum())
    y.append(i)
    j=0
    for valor in dato['p']:
        pxy[j].append(valor)
        j+=1

#Promedios de x, y
x_bar=0
y_bar=0
for i in range(5,16):
    x_bar+=x[i-5]*px[i-5]
for i in range(5,26):
    y_bar+=y[i-5]*py[i-5]

#Varianza
sigma_x=0
sigma_y=0
for i in range(5,16):
    sigma_x+=px[i-5]*(x[i-5]-x_bar)**2
for i in range(5,26):
    sigma_y+=py[i-5]*(y[i-5]-y_bar)**2

#Correlacion, covarianza y coef. correracion
cor=0
cov=0

for i in range(5,16):
    for j in range(5,26):
        cor+=x[i-5]*y[j-5]*pxy[i-5][j-5]
        cov+=(x[i-5]-x_bar)*(y[j-5]-y_bar)*pxy[i-5][j-5]

coef_cor=cov/math.sqrt(sigma_x*sigma_y)
#print(x_bar, y_bar)
#print(sigma_x, sigma_y)
print('Correlacion: ',cor, '\nCovarianza: ', cov, '\nCoeficiente de correlacion: ', coef_cor)

print('\nAjustes')
#Ajuste gaussiano
paramGx, param_covGx = curve_fit(gaussiana, x, px)
print("Gaussian funcion coefficients f(x):", paramGx)
paramGy, param_covGy = curve_fit(gaussiana, y, py)
print("Gaussian funcion coefficients f(y):", paramGy)

#Ajuste de Rayleigh
paramRx, param_covRx = curve_fit(rayleigh, x, px)
print("Rayleigh funcion coefficients f(x):" , paramRx)
paramRy, param_covRy = curve_fit(rayleigh, y, py)
print("Rayleigh funcion coefficients f(y):" , paramRy)

#Ajuste polinomial segundo orden
paramCx, param_covCx = curve_fit(cua, x, px)
print("Funcion cuadrática funcion coefficients f(x):", paramCx)
paramCy, param_covCy = curve_fit(cua, y, py)
print("Funcion cuadrática funcion coefficients f(y):", paramCy)

#Graficar x
pylab.plot(x, px, '.r', label='x')
pylab.plot(x, [rayleigh(x_val, paramRx[0], paramRx[1]) for x_val in x] , '-g', label='rayleigh')
pylab.plot(x, [gaussiana(x_val, paramGx[0], paramGx[1]) for x_val in x] , '-b', label='gaussiana')
pylab.plot(x, [cua(x_val, paramCx[0], paramCx[1], paramCx[2]) for x_val in x] , '-m', label='cuadratica')
pylab.xlabel('x')
pylab.ylabel('Probabilidad')
pylab.title('Ajuste de la variable x')
pylab.legend()
pylab.savefig('ajustex.png')
pylab.show()

#Graficar y
pylab.plot(y, py, '.r', label='y')
pylab.plot(y, [rayleigh(y_val, paramRy[0], paramRy[1]) for y_val in y] , '-g', label='rayleigh')
pylab.plot(y, [gaussiana(y_val, paramGy[0], paramGy[1]) for y_val in y] , '-b', label='gaussiana')
pylab.plot(y, [cua(y_val, paramCy[0], paramCy[1], paramCy[2]) for y_val in y] , '-m', label='cuadratica')
pylab.xlabel('y')
pylab.ylabel('Probabilidad')
pylab.title('Ajuste de la variable y')
pylab.legend()
pylab.savefig('ajustey.png')
pylab.show()

#Errores de las curvas
eRx, eGx, eCx=0,0,0
eRy, eGy, eCy=0,0,0
i=0

for x_val in x:
    eRx+=abs(px[i]-rayleigh(x_val, paramRx[0], paramRx[1]))
    eGx+=abs(px[i]-gaussiana(x_val, paramGx[0], paramGx[1]))
    eCx+=abs(px[i]-cua(x_val, paramCx[0], paramCx[1], paramCx[2]))
    i+=1
i=0
for y_val in y:
    eRy+=abs(py[i]-rayleigh(y_val, paramRy[0], paramRy[1]))
    eGy+=abs(py[i]-gaussiana(y_val, paramGy[0], paramGy[1]))
    eCy+=abs(py[i]-cua(y_val, paramCy[0], paramCy[1], paramCy[2]))
    i+=1

print('Error gaussiana (x, y): ', eGx, eGy, '\nError Rayleigh (x, y): ', eRx, eRy, '\nError polinomio (x, y): ', eCx, eCy)

#Grafico 3D
ejex = np.linspace(5, 15, 100)
ejey = np.linspace(5,25, 100)
ejeX, ejeY = np.meshgrid(ejex, ejey)
ejez=rayleigh(ejeX, paramRx[0], paramRx[1])*rayleigh(ejeY, paramRy[0], paramRy[1])
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(ejeX, ejeY, ejez, color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('p(x, y)');
plt.savefig('3d.png')
plt.show()
