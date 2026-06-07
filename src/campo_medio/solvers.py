import numpy as np
from src.alg_thomas import thomas 
from src.malla import Malla 
from src.campo_medio.config import CampoMedioConfig 
from src.campo_medio.modelo import terminos_reaccion

def inicializar(malla:Malla, config: CampoMedioConfig) -> (np.ndarray, np.ndarray):
    """
    Inicializa las densidades de puntas activas e inactivas.

    La densidad de puntas activas se inicializa con el valor a0 en toda la malla,
    excepto en las fronteras donde se imponen las condiciones de Dirichlet a_in y a_out.

    La densidad de conducto inactivo se inicializa con el valor i0 en toda la malla.

    Parámetros:
        malla:
            Objeto Malla que define la discretización espacial.
        config:
            Parámetros del modelo de campo medio.

    Devuelve:
        Una tupla (A, I) con las densidades iniciales.
    """
    A = np.zeros((malla.nt, malla.nx), dtype=float,)
    I = np.zeros((malla.nt, malla.nx), dtype=float,)

    A[0, :] = config.a0
    I[0, :] = config.i0

    A[0, 0] = config.a_in
    A[0, -1] = config.a_out

    return A, I


def resolver_euler_explicito(malla:Malla, config: CampoMedioConfig) -> (np.ndarray, np.ndarray):
    """
    Resuelve el sistema de campo medio utilizando el método de Euler explícito.

    Parámetros:
        malla:
            Objeto Malla que define la discretización espacial y temporal.
        config:
            Parámetros del modelo de campo medio.

    Devuelve:
        Una tupla (A, I) con las densidades de puntas activas e inactivas en cada paso temporal.
    """
    A, I = inicializar(malla, config)

    
    lambda_ = config.D * malla.dt / malla.dx**2

    for n in range(malla.nt - 1):
        reaccion_a, reaccion_i = terminos_reaccion(A[n, :], I[n, :], config)
        A[n+1, 1:-1] = (
            A[n, 1:-1]
            + lambda_ * (A[n, :-2] - 2 * A[n, 1:-1] + A[n, 2:])
            + malla.dt * reaccion_a[1:-1]
        )

        #I no difunde y se actualiza
        I[n+1, :] = I[n, :] + malla.dt * reaccion_i

        # Condiciones de frontera
        A[n+1, 0] = config.a_in
        A[n+1, -1] = config.a_out

    return A, I


def resolver_imex_cn(malla:Malla, config: CampoMedioConfig) -> (np.ndarray, np.ndarray):
    """
    Resuelve el sistema de campo medio utilizando el método IMEX Crank-Nicolson.

    Parámetros:
        malla:
            Objeto Malla que define la discretización espacial y temporal.
        config:
            Parámetros del modelo de campo medio.
    
    Devuelve:
        Una tupla (A, I) con las densidades de puntas activas e inactivas en cada paso temporal.
    """
    lambda_ = config.D * malla.dt / (2 * malla.dx**2)

    A, I = inicializar(malla, config)

    m = malla.nx - 2
    
    #params Thomas
    d_inf = -lambda_ * np.ones(m-1) #diagonal inferior
    d_prin = (1 + 2*lambda_) * np.ones(m) #diagonal principal
    d_sup = -lambda_ * np.ones(m-1) #diagonal superior

    for n in range(malla.nt-1):
        d = np.zeros(m)

        reaccion_a, reaccion_i = terminos_reaccion(A[n, :], I[n, :], config)

        for j in range(1, malla.nx-1):
            d[j-1] = lambda_ * A[n, j-1] + (1 - 2*lambda_) * A[n, j] + lambda_ * A[n, j+1] + malla.dt * reaccion_a[j]
       
        # corrección por frontera izquierda no nula en el tiempo n+1
        d[0] += lambda_ * config.a_in

        A[n+1,0]= config.a_in #condición de frontera izquierda
        A[n+1,-1] = config.a_out #condición de frontera derecha
        A[n+1, 1:-1] = thomas(d_inf, d_prin, d_sup, d)

        I[n+1, :] = I[n, :] + malla.dt * reaccion_i

    
    return A, I
    