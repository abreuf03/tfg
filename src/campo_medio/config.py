from dataclasses import dataclass

@dataclass
class CampoMedioConfig:
    """
    Parámetros principales del modelo de campo medio.
    
    El sistema considerado es:

        a_t = D a_xx + rb a (1 - (a + i) / n0)

        i_t = re a + (rb / n0) a (a + i)

    con condiciones de Dirichlet para la variable activa:

        a(0, t) = a_in
        a(L, t) = a_out
    """
    D : float = 1.0 #coeficiente de difusión
    rb: float = 0.1 #probabilidad de bifurcación
    re: float = 1.0 #tasa de elongación
    n0: float = 1 #densidad de capacidad local ->valor?

    a_in : float = 1.0 #densidad activa en la frontera de entrada
    a_out : float = 0.0 #densidad activa en la frontera de salida, por defecto 0 pero lo hacemos parámetro para poder experimentar con otras condiciones de frontera

    a0 : float = 0.0 #condición inicial de la variable activa
    i0 : float = 0.0 #condición inicial de la variable inactiva
