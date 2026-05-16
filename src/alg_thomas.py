import numpy as np

#un sistema tridiagonal tiene la forma:
# a_i*x[i-1] + b_i*x[i] + c_i*x[i+1] = d_i
#en este caso, a_i = -lambda_, b_i = 1 + 2*lambda_, c_i = -lambda_ para i=1,...,m-1

#cómo funciona el algoritmo : 
#fase 1 : eliminamos los coeficientes de la diagonal inferior (a_i)
#fase 2 : resolvemos el sistema empezando por la última ecuación y hacia arriba

def thomas(a, b, c, d):
    """
    Resuelve un sistema lineal tridiagonal mediante el algoritmo de Thomas.

    Parámetros:
        a: Diagonal inferior del sistema tridiagonal.
        b: Diagonal principal del sistema tridiagonal.
        c: Diagonal superior del sistema tridiagonal.
        d: Vector del lado derecho del sistema.

    Devuelve:
        Vector solución del sistema tridiagonal.
    """
    n = len(d)

    #trabajamos con copias de los arrays para no modificar los originales
    a_ = a.copy()
    b_ = b.copy()
    c_ = c.copy()
    d_ = d.copy()
    
    #fase 1: eliminación hacia adelante
    for i in range(1, n):
        w = a_[i-1] / b_[i-1]
        b_[i] = b_[i] - w * c_[i-1]
        d_[i] = d_[i] - w * d_[i-1]

    #fase 2: sustitución hacia atrás
    x = np.zeros(n)
    x[-1] = d_[-1] / b_[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d_[i] - c_[i] * x[i+1]) / b_[i]

    return x