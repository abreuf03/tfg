from dataclasses import dataclass

# Este archivo define la clase Punta, que representa cada punta de crecimiento en el modelo BARW.

@dataclass
class Punta:
    """
    Clase que representa una punta de crecimiento en el modelo BARW.
    """

    x: float # posición en x
    y: float # posición en y
    theta: float # dirección de elongación
    id : int # identificador único de la punta
    id_rama : int # identificador de la rama a la que pertenece la punta
    activa : bool = True # indica si la punta está activa (puede elongarse o bifurcarse)
    generacion : int = 0 # generación de la punta (0 para la punta inicial, incrementa con cada bifurcación)
    id_padre : int = None # identificador de la punta madre (None para la punta inicial)