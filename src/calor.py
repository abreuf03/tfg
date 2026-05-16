from src.alg_thomas import thomas
import numpy as np

#para probar : cumple u(0,t) = 0, u(1,t) = 0, u(x,0) = sin(pi*x)
def dato_inicial_seno(x):
    """
    Calcula el dato inicial senoidal para la ecuación del calor.

    Parámetros:
        x: Array con los nodos espaciales.

    Devuelve:
        Array con los valores de sin(pi*x) en los nodos espaciales.
    """
    return np.sin(np.pi * x)

#solucion exacta con este dato inicial
def solucion_exacta_seno(x, t, D):
    """
    Calcula la solución exacta de la ecuación del calor para el dato inicial sin(pi*x).

    Parámetros:
        x: Array con los nodos espaciales.
        t: Tiempo en el que se evalúa la solución.
        D: Coeficiente de difusión.

    Devuelve:
        Array con los valores de la solución exacta en los nodos espaciales.
    """
    return np.exp(-D * np.pi**2 * t) * np.sin(np.pi * x)

def resolver_calor_euler_explicito(malla, D, u0):
    """
    Resuelve la ecuación del calor mediante el método de Euler explícito.

    Parámetros:
        malla: Malla espacial y temporal del problema.
        D: Coeficiente de difusión.
        u0: Dato inicial de la solución.

    Devuelve:
        Matriz U con la aproximación numérica en cada instante y nodo espacial.
    """
    U = np.zeros((malla.nt, malla.nx))
    U[0, :] = u0 #fila cero, todas las columnas -> solución en el instante inicial para todos los puntos espaciales
    lambda_ = D * malla.dt / malla.dx**2 #debe ser menor o igual a 0.5 para estabilidad del método explícito

    for n in range(malla.nt-1):
        U[n+1,0] = 0.0 #condición de frontera izquierda
        U[n+1,-1] = 0.0 #condición de frontera derecha
        for j in range(1, malla.nx-1): #evitar los extremos porque ya los hemos fijado con las condiciones de frontera
            U[n+1, j] = U[n, j] + lambda_ * (U[n, j-1] - 2*U[n, j] + U[n, j+1])
    
    return U

def resolver_calor_crank_nicolson(malla, D, u0):
    """
    Resuelve la ecuación del calor mediante el método de Crank-Nicolson.

    Parámetros:
        malla: Malla espacial y temporal del problema.
        D: Coeficiente de difusión.
        u0: Dato inicial de la solución.

    Devuelve:
        Matriz U con la aproximación numérica en cada instante y nodo espacial.
    """
    lambda_ = D * malla.dt / (2 * malla.dx**2) #en LeVeque aparece sin la D porque toma al principio D=1.0 (lo nota como k, no confundir con  k =delta t)
    #pero aquí la dejamos para dejar claro que resolvemos la ecuación de calor con un coeficiente de difusión D

    U = np.zeros((malla.nt, malla.nx))
    U[0, :] = u0 #fila cero, todas las columnas -> solución en el instante inicial para todos los puntos espaciales

    m = malla.nx - 2 #número de incógnitas en cada paso temporal (excluyendo las condiciones de frontera)


    #params Thomas
    a = -lambda_ * np.ones(m-1) #diagonal inferior
    b = (1 + 2*lambda_) * np.ones(m) #diagonal principal
    c = -lambda_ * np.ones(m-1) #diagonal superior

    #ahora el lado derecho del sistema
    for n in range(malla.nt-1):
        d = np.zeros(m)
        for j in range(1, malla.nx-1):
            d[j-1] = lambda_ * U[n, j-1] + (1 - 2*lambda_) * U[n, j] + lambda_ * U[n, j+1]
        #j-1 porque los nodos interiores van de 1 a nx-2 y b empieza en 0 -> el nodo j=1 está en b[0]

        #finalmente resolvemos el sistema tridiagonal A * U[n+1, 1:-1] = b para cada paso temporal
        U[n+1,0]= 0.0 #condición de frontera izquierda
        U[n+1,-1] = 0.0 #condición de frontera derecha
        #U[n+1, 1:-1] = np.linalg.solve(A, b) #primera implementación -> después usar Thomas
        U[n+1, 1:-1] = thomas(a, b, c, d)
    
    return U