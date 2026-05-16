import numpy as np
from src.alg_thomas import thomas

#ecuación de Fisher-KPP: u_t = D * u_xx + r * u * (1 - u/k)

def funcion_f(u, r, K):
    """
    Calcula el término de reacción logística de la ecuación de Fisher-KPP.

    Parámetros:
        u: Valor o array de valores de la solución.
        r: Tasa de crecimiento.
        K
        : Capacidad de carga.

    Devuelve:
        Valor del término r*u*(1-u/K).
    """
    return r * u * (1 - u/K)

def dato_inicial_fisher_escalon(x, K, x0=2.0): #dato inicial que genera un frente que avanza
    """
    Construye un dato inicial tipo escalón para generar un frente de propagación.

    Parámetros:
        x: Array con los nodos espaciales.
        K: Valor máximo de la población o capacidad de carga.
        x0: Posición inicial del frente.

    Devuelve:
        Array con el dato inicial evaluado en los nodos espaciales.
    """
    u0 = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < x0:
            u0[i] = K
        else:
            u0[i] = 0.0
    return u0

def resolver_fisher_kpp_euler_explicito(malla, D, r, K, u0):
    """
    Resuelve la ecuación de Fisher-KPP mediante el método de Euler explícito.

    Parámetros:
        malla: Malla espacial y temporal del problema.
        D: Coeficiente de difusión.
        r: Tasa de crecimiento del término logístico.
        K: Capacidad de carga y valor de la frontera izquierda.
        u0: Dato inicial de la solución.

    Devuelve:
        Matriz U con la aproximación numérica en cada instante y nodo espacial.
    """
    U = np.zeros((malla.nt, malla.nx))
    U[0, :] = u0

    # Condiciones de frontera iniciales
    U[0, 0] = K
    U[0, -1] = 0.0

    lambda_ = D * malla.dt / malla.dx**2

    for n in range(malla.nt - 1):
        U[n+1, 1:-1] = (
            U[n, 1:-1]
            + lambda_ * (U[n, :-2] - 2 * U[n, 1:-1] + U[n, 2:])
            + malla.dt * funcion_f(U[n, 1:-1], r, K)
        )

        # Condiciones de frontera
        U[n+1, 0] = K
        U[n+1, -1] = 0.0

    return U


def posicion_frente_por_nivel(x, u, nivel):
    """
    Calcula la posición del frente como el punto donde la solución cruza un nivel dado.

    Parámetros:
        x: Array con los nodos espaciales.
        u: Array con los valores de la solución en un instante fijo.
        nivel: Valor de referencia usado para localizar el frente.

    Devuelve:
        Posición interpolada del frente si existe cruce; en caso contrario, None.
    """
    for j in range(len(x) - 1):
        u_izq = u[j]
        u_der = u[j + 1]

        # buscamos un cruce descendente o exacto
        if (u_izq >= nivel and u_der <= nivel) or (u_izq <= nivel and u_der >= nivel):
            if u_der == u_izq:
                return x[j]
            alpha = (nivel - u_izq) / (u_der - u_izq)
            return x[j] + alpha * (x[j + 1] - x[j])

    return None

def resolver_fisher_kpp_cn_semimplicito(malla, D, r, K, u0):
    """
    Resuelve la ecuación de Fisher--KPP mediante un esquema semi-implícito de tipo IMEX.

    La parte difusiva se trata con Crank--Nicolson y el término de reacción
    se evalúa explícitamente en el instante temporal anterior.


    Parámetros:
        malla: Malla espacial y temporal del problema.
        D: Coeficiente de difusión.
        r: Tasa de crecimiento del término logístico.
        K: Capacidad de carga y valor de la frontera izquierda.
        u0: Dato inicial de la solución.

    Devuelve:
        Matriz U con la aproximación numérica en cada instante y nodo espacial.
    """
    
    lambda_ = D * malla.dt / (2 * malla.dx**2)
   
    U = np.zeros((malla.nt, malla.nx))
    U[0, :] = u0
    U[0, 0] = K
    U[0, -1] = 0.0

    m = malla.nx - 2
    
    #params Thomas
    a = -lambda_ * np.ones(m-1) #diagonal inferior
    b = (1 + 2*lambda_) * np.ones(m) #diagonal principal
    c = -lambda_ * np.ones(m-1) #diagonal superior

    for n in range(malla.nt-1):
        d = np.zeros(m)
        for j in range(1, malla.nx-1):
            d[j-1] = lambda_ * U[n, j-1] + (1 - 2*lambda_) * U[n, j] + lambda_ * U[n, j+1] + malla.dt * funcion_f(U[n, j], r, K)
       
        # corrección por frontera izquierda no nula en el tiempo n+1
        d[0] += lambda_ * K

        U[n+1,0]= K #condición de frontera izquierda
        U[n+1,-1] = 0.0 #condición de frontera derecha
        #U[n+1, 1:-1] = np.linalg.solve(A, b) #primera implementación -> después usar Thomas
        U[n+1, 1:-1] = thomas(a, b, c, d)
    
    return U