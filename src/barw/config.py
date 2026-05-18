from dataclasses import dataclass
from math import pi

#en este archivo incluiremos todos los parametros de configuracion
#se toman del apartado "Mammary gland parameters and numerical simulations" del paper de Hannezo et. al. (2017) 

@dataclass
class BARWConfig:
    """
    Parámetros principales del modelo BARW.
    """

    # Parámetro del dominio espacial
    Lx : float = 280.0
    Ly : float = 150.0

    #Parámetros de la dinámica
    v : float = 1.0 #velocidad de elongación
    rb : float = 0.1 #tasa de bifurcación
    Ra : float = 3 #radio de aniquilación
    long_paso : float = 1.0 #longitud del paso de los random walkers
    tiempo_paso : float = 1.0 #tiempo entre pasos de los random walkers
    ang_amplitud : float = pi/10 #amplitud del ángulo de difusión de la dirección de elongación
    angulo_bifurcacion: float = pi / 6

    #Parámetros del dominio temporal
    tiempo_total : float = 1000.0 #tiempo total de la simulación

    #Condición inicial
    x0 : float = 0.0 #posición inicial en x
    theta0 : float = 0.0 #dirección inicial de elongación

    #Reproducibilidad
    semilla : int = 42 #semilla para la generación de números aleatorios

    #Máximos computacionales
    max_puntas : int = 100000 #número máximo de puntas para evitar sobrecarga computacional
    max_pasos : int = 1000000 #número máximo de pasos para evitar sobrecarga computacional

    #Gestionar autoaniquilación de puntas recién bifurcadas
    pasos_exclusion_aniquilacion : int = 6 #número de pasos durante los cuales una punta recién bifurcada no puede ser aniquilada por estar cerca de su madre


