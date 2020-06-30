# Parte a
Se desea encontrar la curva de mejor ajuste. Para ello se ajustan los datos a tres modelos diferentes:
  - Distribución gaussiana
  - Distribución de Rayleigh
  - Polinomio cuadrático ax^2+bx+c

Los parámetros para la distribución de 'x' son los siguientes:
  - Distribución gaussiana: mu= 9.9 y sigma=3.3 
  - Distribución de Rayleigh: a=3.96 y b=55.9
  - Polinomio cuadrático ax^2+bx+c: a=-0.0024951, b=0.04930901, c=-0.12771399

Los parámetros para la distribución de 'y' son los siguientes:
  - Distribución gaussiana: mu= 15.08 y sigma=6.03 
  - Distribución de Rayleigh: a=3.88 y b=199
  - Polinomio cuadrático ax^2+bx+c: a=-0.00037869,  b=0.01141883, c=-0.02456999

Si definimos el error como el valor absoluto de la resta del valor real y el valor dado por la función de la curva de mejor ajuste (abs[Px-f(x)]) donde Px está dado por los datos y f son las funciones obtenidas anteriormente.

Los errores son:
  - Distribución gaussiana: ex= 0.12 y ey=0.18 
  - Distribución de Rayleigh: ex=0.145 y ey=0.2
  - Polinomio cuadrático: ex=0.14 y ey=0.21

La distribución gaussiana es la que tiene menor error tanto para 'x' como para 'y', entonces es la que se ajusta mejor (menos mal).

# Parte b
Si se asume independencia de los datos, entonces f(x, y)=f(x)f(y). Si se toman ambas funciones como las gaussianas definidas anteriormente, entonces la función de densidad conjunta es: 

$$\frac{1}{2\pi \sigma_x\sigma _y}e^{-\frac{1}{2}\left ( \left ( \frac{x-\mu_x}{\sigma_x} \right )^2+\left ( \frac{y-\mu_y}{\sigma_y} \right )^2 \right )}$$

# Parte c
La correlación es el grado en el cual dos o más cantidades están linealmente
asociadas. Para las variables dadas, la correlación es 149.54. Entonces las variables están correlacionadas.

La covarianza indica la variación de dos variables aleatorias respecto a sus medias. Para las variables dadas, la covarianza es 0.06773050531225652. 

El coeficiente de correlación indica si existe una correlación o relación de proporcionalidad entre las variables. Para las variables dadas, el coeficiente de correlación es 0.004552904974476006. Entonces la relación de proporcionalidad de las variables es muy pequeña.

# Parte d
Las funciones de densidad marginales se pueden ver en las figuras 'ajustex.png' y 'ajustey.png'. La función de densidad conjunta se puede ver en '3d.png'
