import numpy as np

from .config import BARWConfig
from .punta import Punta
from .busqueda_espacial import ExhaustivaIndices, KDTreeIndices, QuadTreeNode, QuadTreeIndices

class SimulacionBARW:
    """
    Clase principal para ejecutar la simulación del modelo BARW.
    """

    def __init__(self, config: BARWConfig, metodo_busqueda:int=0):
        self.config = config
        # 0: Exhaustiva, 1: KDTree, 2: QuadTree
        self.metodo_busqueda = metodo_busqueda

        self.puntas = [] # lista de puntas activas
        self.ramas = [] # lista de ramas (cada rama es una lista de puntas)
        self.conducto = []
        
        self.rng = np.random.default_rng(self.config.semilla)

        if self.metodo_busqueda == 0:
            self.busqueda_espacial = ExhaustivaIndices()

        elif self.metodo_busqueda == 1:
            self.busqueda_espacial = KDTreeIndices()

        elif self.metodo_busqueda == 2:
            self.busqueda_espacial = QuadTreeIndices(
                x_min=0.0,
                x_max=self.config.Lx,
                y_min=0.0,
                y_max=self.config.Ly,
                capacidad=8,
                profundidad_maxima=15,
            )

        else:
            raise ValueError(
                "metodo_busqueda debe ser 0 (exhaustiva), "
                "1 (KDTree) o 2 (QuadTree)."
            )
        
        self.siguiente_id_punta = 0 # contador para asignar identificadores únicos a las puntas
        self.siguiente_id_rama = 0 # contador para asignar identificadores únicos a las ramas

        self.contador_pasos = 0 # contador de pasos de la simulación
        self.contador_bifurcaciones = 0 # contador de bifurcaciones realizadas
        self.contador_terminaciones = 0 # contador de terminaciones por aniquilación
        self.contador_colisiones = 0 #terminaciones por colisión
        self.contador_salidas = 0 #terminacios por salida de dominio

        #para comprobar aniquilación debemos guardar en qué paso se da la bifurcación y sólo
        #ignorar en un número limitado de pasos
        #paso_bifurcacion[5]=20, la rama 5 nacio de una bifurcacion en el paso 20
        self.paso_bifurcacion: dict[int, int] = {}
        # guarda para cada rama hija, cual es su madre : rama_padre[5]=2, la hija 5 nace de la rama 2
        self.rama_padre: dict[int, int | None] = {}
    

    def inicializar(self):
        """
        Inicializa la simulación con una única punta activa situada
        en el lado izquierdo del dominio.
        """

        x0 = self.config.x0
        y0 = self.config.Ly / 2.0
        theta0 = self.config.theta0

        punta_inicial = Punta(x=x0, y=y0, theta=theta0, 
                              id=self.siguiente_id_punta, id_rama=self.siguiente_id_rama,
                              generacion=0, activa=True)
        self.puntas.append(punta_inicial)
        self.busqueda_espacial.agregar_punto(x0, y0, self.siguiente_id_rama,0)
        self.ramas.append([punta_inicial]) # cada rama comienza con una única punta
        self.siguiente_id_punta += 1
        self.siguiente_id_rama += 1
        self.paso_bifurcacion[punta_inicial.id_rama]=0
        self.rama_padre[punta_inicial.id_rama]= None

        if self.metodo_busqueda == 1:
            self.busqueda_espacial.construir_kdtree()
        
    
    def mover_punta(self, punta: Punta):
        """
        Mueve una punta activa según la dinámica de elongación y difusión.

        Parámetros:
            punta (Punta): La punta a mover.
        
        Devuelve:
            x_anterior (float): La posición x anterior de la punta.
            y_anterior (float): La posición y anterior de la punta.
            x_nueva (float): La nueva posición x de la punta después del movimiento.
            y_nueva (float): La nueva posición y de la punta después del movimiento.
        """
        #guardamos los valores anteriores para poder revertir el movimiento en caso de aniquilación
        x_anterior = punta.x
        y_anterior = punta.y

        # Elongación en la dirección actual
        dx = self.config.long_paso * np.cos(punta.theta)
        dy = self.config.long_paso * np.sin(punta.theta)

        # Difusión aleatoria del ángulo de elongación
        dtheta = self.rng.uniform(-self.config.ang_amplitud, self.config.ang_amplitud) #en el artículo se menciona que el ángulo de difusión es aleatorio con una amplitud de pi/10, por lo que se toma un valor aleatorio entre -pi/10 y pi/10

        # Actualizar posición y dirección de la punta
        punta.x += dx
        punta.y += dy
        punta.theta += dtheta

        return x_anterior, y_anterior, punta.x, punta.y


    def fuera_de_limites(self, punta: Punta):
        """
        Verifica si una punta ha salido de los límites del dominio.

        Parámetros:
            punta (Punta): La punta a verificar.
        
        Devuelve:
            bool: True si la punta está fuera de los límites, False en caso contrario.
        """
        return (punta.x < 0 or punta.x > self.config.Lx or
                punta.y < 0 or punta.y > self.config.Ly)
    

    def crear_bifurcacion(self, punta_madre: Punta, paso:int):
        """
        Crea una bifurcación a partir de una punta activa.

        Parámetros:
            punta_madre (Punta): La punta madre desde la cual crear la bifurcación.
        
        Devuelve:
            Punta: La nueva punta creada por la bifurcación.
        """
        signo = self.rng.choice([-1, 1]) # la bifurcación puede ocurrir a la izquierda o a la derecha de la dirección de elongación
        angulo_bifurcacion = signo * self.config.angulo_bifurcacion

        nueva_punta = Punta(
            x=punta_madre.x,
            y=punta_madre.y,
            theta=punta_madre.theta + angulo_bifurcacion, # la nueva punta se bifurca con un ángulo estocástico
            id=self.siguiente_id_punta,
            id_rama=self.siguiente_id_rama, # la nueva punta pertenece a una nueva rama
            activa=True,
            generacion=punta_madre.generacion + 1, # la generación de la nueva punta es una más que la generación de la punta madre
            id_padre=punta_madre.id,
            edad=0,
            id_rama_padre=punta_madre.id_rama
        )

        self.rama_padre[self.siguiente_id_rama]=punta_madre.id_rama
        self.paso_bifurcacion[self.siguiente_id_rama]=paso
        self.siguiente_id_punta += 1
        self.siguiente_id_rama += 1
        

        return nueva_punta
    
    
    def distancia_punto_segmento(self,px, py, x0, y0, x1, y1):
        """
        Calcula la distancia entre un punto P=(px,py)
        y el segmento que une A=(x0,y0) con B=(x1,y1).
        """

        p = np.array([px, py])
        a = np.array([x0, y0])
        b = np.array([x1, y1])

        ab = b - a
        ap = p - a

        norma_ab2 = np.dot(ab, ab)

        if norma_ab2 == 0:
            return np.linalg.norm(p - a)

        t = np.dot(ap, ab) / norma_ab2
        t = np.clip(t, 0.0, 1.0)

        proyeccion = a + t * ab

        return np.linalg.norm(p - proyeccion)
    
    def exclusion_local(self, punta:Punta, candidato:int, paso_actual:int)->bool:
        #criterio de colision:
        #candidato de mi misma rama ->ignoro solo si es reciente
        #candidato madre/hija -> ignoro durante el periodo de exclusion de bifurcacion
        #candidato de otra rama -> terminacion

        id_rama_candidato = self.busqueda_espacial.ramas_ids[candidato]
        num_paso = self.busqueda_espacial.pasos[candidato]

        if id_rama_candidato==punta.id_rama:
            return (paso_actual-num_paso <= self.config.pasos_exclusion_propia)
        

        madre_hija = (self.rama_padre.get(punta.id_rama) == id_rama_candidato) or (
            self.rama_padre.get(id_rama_candidato)==punta.id_rama)
        
        if madre_hija:

            paso_bifurcacion = max(
                self.paso_bifurcacion.get(punta.id_rama,0),
                self.paso_bifurcacion.get(id_rama_candidato,0)
                #de esta forma nos da el paso en el que nace la hija, que siempre es > que el de la madre
            )

            periodo_exclusion = (paso_actual-paso_bifurcacion <= self.config.pasos_exclusion_aniquilacion)
            
            #esta sola NO es suficiente, porque va a evitar durante 6 pasos que haya colisiones entre madre e hija
            #pero si que deberiamos permitir colisiones con conductos mas antiguos de la madre
            #es decir, solo queremos evitar los pasos cercanos a la bifurcacion, entonces:
           
            cercano_a_bifrucacion = (num_paso >= 
                                     paso_bifurcacion - self.config.pasos_exclusion_propia)

            return periodo_exclusion and cercano_a_bifrucacion
        else:
            return False

    
    def colision_admisible(self, punta:Punta, cercanas:list[int], paso_actual:int)->bool:

        for indice in cercanas : 
            if not self.exclusion_local(punta,indice,paso_actual):
                return True
            
        return False

    
    def paso(self):
        """
        Ejecuta un paso temporal de la simulación.
        """
        nuevas_puntas = []
        puntos_nuevos = []
        
        paso_actual = self.contador_pasos +1

        for punta in self.puntas:
            #hay_colision = False
            if not punta.activa:
                continue

            # 1. Mover la punta
            x_anterior, y_anterior, x_nueva, y_nueva = self.mover_punta(punta)

            # 2. Verificar si la punta ha salido de los límites del dominio
            if self.fuera_de_limites(punta):
                punta.activa = False
                self.contador_terminaciones += 1
                self.contador_salidas+=1
                punta.x = x_anterior
                punta.y = y_anterior
                continue

            # 3. Buscar aniquilación contra la red ya existente
            puntas_cercanas = self.busqueda_espacial.buscar_puntas_cercanas(x_nueva,y_nueva,self.config.Ra)
            

            # 6. Actualizar edad
            punta.edad += 1

            # 7. Si había colisión, la punta termina
            if self.colision_admisible(punta,puntas_cercanas,paso_actual):
                punta.activa = False
                self.contador_terminaciones += 1
                self.contador_colisiones+=1
                punta.x = x_anterior
                punta.y = y_anterior
                continue
            
                        
            # 5. Guardar el punto para añadirlo después al índice espacial
            puntos_nuevos.append((x_nueva, y_nueva, punta.id_rama,paso_actual))

            # 8. Bifurcación estocástica
            if self.rng.random() < self.config.pb:
                nueva_punta = self.crear_bifurcacion(punta,paso=paso_actual)
                nuevas_puntas.append(nueva_punta)
                self.contador_bifurcaciones += 1
                
            
            # 4. Guardar el segmento generado
            self.conducto.append(
                (x_anterior, y_anterior, x_nueva, y_nueva, punta.id_rama)
            )





        # 9. Añadir todos los puntos nuevos al índice espacial
        for x, y, id_rama, paso in puntos_nuevos:
            self.busqueda_espacial.agregar_punto(x, y, id_rama,paso)

        # 10. Reconstruir el KDTree solo una vez por paso temporal
        if self.metodo_busqueda == 1:
            self.busqueda_espacial.construir_kdtree()

        # 11. Añadir las puntas nuevas
        self.puntas.extend(nuevas_puntas)

        self.contador_pasos += 1

    def ejecutar(self):
        """
        Ejecuta la simulación del modelo BARW hasta alcanzar el tiempo total configurado
        o hasta que no queden puntas activas.

        Devuelve:
        Un diccionario con el historial de la simulación, incluyendo el tiempo, número de puntas activas, número de bifurcaciones, número de terminaciones y número total de puntas en cada paso.
        """
        self.inicializar()

        historial={
            "tiempo": [],
            "num_puntas_activas": [],
            "num_bifurcaciones": [],
            "num_terminaciones": [],
            "num_puntas_totales": [],
            "x_max": [],
            "num_colisiones": [],
            "num_salidas_frontera": [],
        }

        num_pasos = int(self.config.tiempo_total / self.config.tiempo_paso)
        num_pasos = min(num_pasos,self.config.max_pasos)

        for paso in range(num_pasos):   


            self.paso() #aquí se avanza un paso, por eso para el tiempo es paso+1
            tiempo = (paso+1) * self.config.tiempo_paso #antes guardamos paso 0 en t=1

            

            if self.conducto:
                x_max = max(max(seg[0], seg[2]) for seg in self.conducto)
            else:
                x_max = 0.0

            puntas_activas = sum(p.activa for p in self.puntas)

            # Guardar datos para análisis posterior
            historial["tiempo"].append(tiempo)
            historial["num_puntas_activas"].append(puntas_activas)
            historial["num_bifurcaciones"].append(self.contador_bifurcaciones)
            historial["num_terminaciones"].append(self.contador_terminaciones)
            historial["num_puntas_totales"].append(len(self.puntas))
            historial["x_max"].append(x_max)
            historial["num_colisiones"].append(self.contador_colisiones)
            historial["num_salidas_frontera"].append(self.contador_salidas)

            if puntas_activas == 0:
                print(f"Simulación terminada en el paso {paso+1} (tiempo={tiempo:.2f}) - No quedan puntas activas.")
                break

            

            if len(self.puntas) > self.config.max_puntas:
                print(
                    f"Simulación terminada en el paso {paso+1} "
                    f"(tiempo={tiempo:.2f}) - Se alcanzó el máximo de puntas."
                )
                break

        #print("Colisiones solo por segmentos:", self.colisiones_segmento)
        #print("Colisiones solo por puntos:", self.colisiones_puntos)
        #print("Colisiones por ambas:", self.colisiones_ambas)

        return {
            "historial": historial,
            "conducto": self.conducto,
            "puntas": self.puntas,
            "config": self.config
        }