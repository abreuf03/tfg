import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass

# En este archivo implementamos las funciones de búsqueda espacial 
# para una búsqueda exhaustiva y para una búsqueda eficiente utilizando KDTree.

class ExhaustivaIndices:
    """
    Clase para realizar una búsqueda exhaustiva de puntas dentro del radio de aniquilación.
    """

    def __init__(self):
        self.puntos = []
        self.ramas_ids = []
        self.pasos = []

    def agregar_punto(self, x,y, id_rama, paso):
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
            #esta lógica ignora la rama completa, buscamos implementar exclusión local
            #devolvemos simplemente los vecinos cercanos y en otra función en el simulador estudiamos si se tienen en cuenta o no para la terminación
            #if excluir_id_rama is not None and self.ramas_ids[i] == excluir_id_rama: #si no se añade casi todas las ramas se aniquiliarían por estar cercanas a su propia rama
            #    continue
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
        #if excluir_id_rama is not None:
        #    indices = [i for i in indices if self.ramas_ids[i] != excluir_id_rama]
        return indices
    
@dataclass
class QuadTreeNode:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    indices: list[int]
    hijos : list
    capacidad: int = 4  # Capacidad máxima de puntos por nodo antes de subdividir
    max_profundidad: int = 10  # Profundidad máxima del QuadTree
    profundidad: int = 0  # Profundidad actual del nodo



    def contiene(self, x, y) -> bool: 
        """
        Verifica si un punto (x, y) está dentro de la región del nodo.

        Parámetros:
            x (float): Coordenada x del punto.
            y (float): Coordenada y del punto.

        Devuelve:
            bool: True si el punto está dentro de la región del nodo, False en caso contrario.
        """
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max
    
    def interseca(self, x_min, x_max, y_min, y_max) -> bool: #equivale a RECTANGULO_OVERLAPS_REGION en el paper
        """
        Verifica si la región del nodo interseca con un rectángulo definido por (x_min, x_max, y_min, y_max).

        Parámetros:
            x_min (float): Coordenada mínima en x del rectángulo.
            x_max (float): Coordenada máxima en x del rectángulo.
            y_min (float): Coordenada mínima en y del rectángulo.
            y_max (float): Coordenada máxima en y del rectángulo.
        
        Devuelve:
            bool: True si hay intersección, False en caso contrario.
        """
        return  (x_max >= self.x_min and x_min <= self.x_max and y_max >= self.y_min and y_min <= self.y_max)
        

class QuadTreeIndices:
    """
    Clase para realizar una búsqueda eficiente de puntas dentro del radio de aniquilación utilizando QuadTree.
    """

    def __init__(self,
        x_min: float, 
        x_max: float,
        y_min: float, 
        y_max: float,
        capacidad: int = 8,
        profundidad_maxima: int = 10
    ):
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("Los límites del QuadTree son inválidos.")
        if capacidad <= 0:
            raise ValueError("La capacidad del QuadTree debe ser mayor que cero.")
        if profundidad_maxima <= 0:
            raise ValueError("La profundidad máxima del QuadTree debe ser mayor que cero.")
        
        self.capacidad = capacidad
        self.profundidad_maxima = profundidad_maxima
        
        self.root = QuadTreeNode(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            indices=[],
            hijos=None,
            capacidad=capacidad,
            max_profundidad=profundidad_maxima,
            profundidad=0,
        )

        self.puntos: list[tuple[float, float]] = []
        self.ramas_ids: list[int] = []
        self.pasos: list[int] = []


    def agregar_punto(self, x, y, id_rama: int, paso):
        """
        Añade un punto de la red ya depositada.

        Parámetros:
            x (float): Coordenada x del punto.
            y (float): Coordenada y del punto.
            id_rama (int): Identificador de la rama a la que pertenece el punto.
        """
        if not self.root.contiene(x, y):
            raise ValueError(f"El punto ({x}, {y}) está fuera de los límites del QuadTree.")
        
        indice = len(self.puntos)
        self.puntos.append((x, y))
        self.ramas_ids.append(id_rama)
        self.pasos.append(paso)
        self._insertar(self.root, indice)

    def _insertar(self, nodo: QuadTreeNode, indice: int):
        """
        Inserta un índice en el nodo del QuadTree.

        Parámetros:
            nodo (QuadTreeNode): Nodo actual del QuadTree.
            indice (int): Índice del punto a insertar.
        """

        # Si es una hoja, intentamos almacenar aquí el punto.
        if nodo.hijos is None:
            if (
                len(nodo.indices) < nodo.capacidad
                or nodo.profundidad >= nodo.max_profundidad
            ):
                nodo.indices.append(indice)
                return

            # La hoja está llena: se subdivide.
            indices_anteriores = nodo.indices
            nodo.indices = []

            self._subdividir(nodo)

            # Redistribuimos los puntos que estaban en la hoja.
            for indice_anterior in indices_anteriores:
                hijo = self._obtener_hijo(nodo, indice_anterior)
                self._insertar(hijo, indice_anterior)

        # Insertamos el nuevo punto en un único hijo.
        hijo = self._obtener_hijo(nodo, indice)
        self._insertar(hijo, indice)

    def _subdividir(self, nodo: QuadTreeNode):
        """
        Subdivide un nodo en cuatro hijos.

        Parámetros:
            nodo (QuadTreeNode): Nodo a subdividir.
        """
        x_mid = (nodo.x_min + nodo.x_max) / 2
        y_mid = (nodo.y_min + nodo.y_max) / 2

        sig_profundidad = nodo.profundidad + 1

        # Orden: SO, SE, NO, NE
        nodo.hijos = [
            # SO
            QuadTreeNode(
                x_min=nodo.x_min,
                x_max=x_mid,
                y_min=nodo.y_min,
                y_max=y_mid,
                indices=[],
                hijos=None,
                capacidad=nodo.capacidad,
                max_profundidad=nodo.max_profundidad,
                profundidad=sig_profundidad,
            ),

            # SE
            QuadTreeNode(
                x_min=x_mid,
                x_max=nodo.x_max,
                y_min=nodo.y_min,
                y_max=y_mid,
                indices=[],
                hijos=None,
                capacidad=nodo.capacidad,
                max_profundidad=nodo.max_profundidad,
                profundidad=sig_profundidad,
            ),

            # NO
            QuadTreeNode(
                x_min=nodo.x_min,
                x_max=x_mid,
                y_min=y_mid,
                y_max=nodo.y_max,
                indices=[],
                hijos=None,
                capacidad=nodo.capacidad,
                max_profundidad=nodo.max_profundidad,
                profundidad=sig_profundidad,
            ),

            # NE
            QuadTreeNode(
                x_min=x_mid,
                x_max=nodo.x_max,
                y_min=y_mid,
                y_max=nodo.y_max,
                indices=[],
                hijos=None,
                capacidad=nodo.capacidad,
                max_profundidad=nodo.max_profundidad,
                profundidad=sig_profundidad,
            ),
        ]

        
    
    def _obtener_hijo(self, nodo: QuadTreeNode, indice: int) -> QuadTreeNode: #equivale a COMPARE en el paper
        """
        """
        if nodo.hijos is None:
            raise ValueError("El nodo no está subdividido.")
        
        x,y = self.puntos[indice]

        x_medio = (nodo.x_min + nodo.x_max) / 2
        y_medio = (nodo.y_min + nodo.y_max) / 2

        derecha = x >= x_medio
        arriba = y >= y_medio

        if arriba:
            if derecha:
                return nodo.hijos[3]  # NE
            else:
                return nodo.hijos[2]  # NO
        else:
            if derecha:
                return nodo.hijos[1]  # SE
            else:
                return nodo.hijos[0]  # SO
    
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
        candidatos = []

        self._buscar_en_rectangulo(
            nodo=self.root,
            x_min=x - Ra,
            x_max=x + Ra,
            y_min=y - Ra,
            y_max=y + Ra,
            candidatos=candidatos,
        )

        radio2 = Ra ** 2
        cercanas = []

        for i in candidatos:
            #if excluir_id_rama is not None and self.ramas_ids[i] == excluir_id_rama:
            #    continue
            px, py = self.puntos[i]
            distancia2 = (px - x) ** 2 + (py - y) ** 2
            if distancia2 <= radio2:
                cercanas.append(i)

        return cercanas
    
    def _buscar_en_rectangulo(self, nodo: QuadTreeNode, x_min, x_max, y_min, y_max, candidatos): #equivalente a REGIONSEARCH en el paper
        """
        Busca en el QuadTree los puntos que están dentro de un rectángulo definido por (x_min, x_max, y_min, y_max).

        Parámetros:
            nodo (QuadTreeNode): Nodo actual del QuadTree.
            x_min (float): Coordenada mínima en x del rectángulo.
            x_max (float): Coordenada máxima en x del rectángulo.
            y_min (float): Coordenada mínima en y del rectángulo.
            y_max (float): Coordenada máxima en y del rectángulo.
            candidatos (list): Lista donde se almacenarán los índices de los puntos encontrados.
        """
        if not nodo.interseca(x_min, x_max, y_min, y_max):
            return  # No hay intersección con este nodo

        # Si es una hoja, agregamos sus índices a la lista de candidatos
        if nodo.hijos is None:
            candidatos.extend(nodo.indices)
            return

        # Si no es una hoja, buscamos recursivamente en los hijos
        for hijo in nodo.hijos:
            self._buscar_en_rectangulo(hijo, x_min, x_max, y_min, y_max, candidatos)

