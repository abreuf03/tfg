import numpy as np

from src.campo_medio.config import CampoMedioConfig

def terminos_reaccion(a:np.ndarray, i:np.ndarray, config: CampoMedioConfig) -> (np.ndarray, np.ndarray):
    """
    Calcula los términos de reacción del sistema de campo medio.

    El término correspondiente a la densidad activa es:

        F(a, i) = rb * a * (1 - (a + i) / n0)

    El término correspondiente a la densidad inactiva es:

        G(a, i) = re * a + (rb / n0) * a * (a + i)

    La difusión de a no se incluye en esta función, ya que se
    discretiza por separado en los métodos numéricos.

    Parámetros:
        a:
            Densidad de puntas activas.
        i:
            Densidad de conducto inactivo.
        config:
            Parámetros del modelo de campo medio.

    Devuelve:
        Una tupla (reaccion_a, reaccion_i).
    """

    ocupacion = a + i

    reaccion_a = (config.rb * a * (1.0 - ocupacion / config.n0))

    reaccion_i = (config.re * a + (config.rb / config.n0) * a * ocupacion)

    return reaccion_a, reaccion_i