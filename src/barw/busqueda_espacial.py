import numpy as np
from scipy.spatial import cKDTree

# En este archivo implementamos las funciones de búsqueda espacial 
# para una búsqueda exhaustiva y para una búsqueda eficiente utilizando KDTree.

class ExhaustivaIndices:
    """
    Clase para realizar una búsqueda exhaustiva de puntas dentro del radio de aniquilación.
    """

    def __init__(self):
        self.puntos = []
        self.ramas_ids = []

    def agregar_punto(self, x,y, id_rama):
        """
        Añade un punto de la red ya depositada.

        Parámetros:
            x (float): Coordenada x del punto.
            y (float): Coordenada y del punto.
            id_rama (int): Identificador de la rama a la que pertenece el punto.
        """
        self.puntos.append((x, y))
        self.ramas_ids.append(id_rama)
    
    def buscar_puntas_cercanas(self, x, y, Ra, excluir_id_rama=None):
        """
        Devuelve los índices de los puntos que están a distancia menor que Ra.

        Si excluir_rama no es None, se ignoran los puntos pertenecientes
        a esa misma rama.

        Parámetros:
            x (float): Coordenada x del punto de referencia.
            y (float): Coordenada y del punto de referencia.
            Ra (float): Radio de aniquilación.
            excluir_id_rama (int, opcional): Identificador de la rama a excluir en la búsqueda.
        
        Devuelve:
            Lista de índices de los puntos cercanos dentro del radio de aniquilación.
        """
        cercanas = []
        for i, (px, py) in enumerate(self.puntos):
            if excluir_id_rama is not None and self.ramas_ids[i] == excluir_id_rama: #si no se añade casi todas las ramas se aniquiliarían por estar cercanas a su propia rama
                continue
            distancia_2 = ((px - x) ** 2 + (py - y) ** 2)
            if distancia_2 <= Ra ** 2:
                cercanas.append(i)
        return cercanas

class KDTreeIndices:
    """
    Clase para realizar una búsqueda eficiente de puntas dentro del radio de aniquilación utilizando cKDTree.
    """

    def __init__(self):
        self.puntos = []
        self.ramas_ids = []
        self.kdtree = None
        self.pasos = []

    def agregar_punto(self, x,y, id_rama, paso=None):
        """
        Añade un punto de la red ya depositada.

        Parámetros:
            x (float): Coordenada x del punto.
            y (float): Coordenada y del punto.
            id_rama (int): Identificador de la rama a la que pertenece el punto.
        """
        self.puntos.append((x, y))
        self.ramas_ids.append(id_rama)
        self.pasos.append(paso)
    
    def construir_kdtree(self):
        """
        Construye o reconstruye el árbol k-d.
        """
        if self.puntos:
            self.kdtree = cKDTree(self.puntos)
        else:
            self.kdtree = None

    def buscar_puntas_cercanas(self, x, y, Ra, excluir_id_rama=None):
        """
        Devuelve los índices de los puntos que están a distancia menor que Ra.
        Si excluir_rama no es None, se ignoran los puntos pertenecientes
        a esa misma rama.

        Parámetros:
            x (float): Coordenada x del punto de referencia.
            y (float): Coordenada y del punto de referencia.
            Ra (float): Radio de aniquilación.
            excluir_id_rama (int, opcional): Identificador de la rama a excluir en la búsqueda.
            
        Devuelve:
            Lista de índices de los puntos cercanos dentro del radio de aniquilación.
        """
        if self.kdtree is None:
            return []
        indices = self.kdtree.query_ball_point((x, y), Ra)
        if excluir_id_rama is not None:
            indices = [i for i in indices if self.ramas_ids[i] != excluir_id_rama]
        return indices