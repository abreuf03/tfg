import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

rng = np.random.default_rng(42)
dimension = 2
radio = 0.02
repeticiones = 3
tamanos = [100, 1000, 10000, 100000]


def generar_datos(N, dimension, rng):
    """
    Genera los puntos almacenados y los puntos de consulta para el benchmark.

    Parámetros:
        N: Número de puntos almacenados y de puntos de consulta.
        dimension: Dimensión del espacio.
        rng: Generador aleatorio de NumPy.

    Devuelve:
        puntos: Array de tamaño (N, dimension) con los puntos almacenados.
        consultas: Array de tamaño (N, dimension) con los puntos de consulta.
    """
    puntos = rng.random((N, dimension))
    consultas = rng.random((N, dimension))
    return puntos, consultas



def guardar_resultados(resultados, ruta_salida):
    """
    Guarda los resultados del benchmark en un archivo CSV.

    Parámetros:
        resultados: Lista de diccionarios con los tiempos medidos.
        ruta_salida: Ruta del archivo CSV de salida.

    Devuelve:
        None.
    """
    df = pd.DataFrame(resultados)
    df.to_csv(ruta_salida, index=False)


def generar_figura_loglog(resultados, ruta_figura):
    """
    Genera una figura log-log comparando los tiempos de búsqueda exhaustiva y cKDTree.

    Parámetros:
        resultados: Lista de diccionarios con los tiempos medidos.
        ruta_figura: Ruta donde se guardará la figura.

    Devuelve:
        None.
    """
    N = np.array([res["N"] for res in resultados])
    tiempos_exhaustivo = np.array([res["tiempo_exhaustivo"] for res in resultados])
    tiempos_kdtree = np.array([res["tiempo_kdtree"] for res in resultados])

    mascara_exhaustivo = np.isfinite(tiempos_exhaustivo)

    plt.figure(figsize=(8, 6))

    plt.loglog(
        N[mascara_exhaustivo],
        tiempos_exhaustivo[mascara_exhaustivo],
        marker="o",
        label="Búsqueda exhaustiva",
    )

    plt.loglog(
        N,
        tiempos_kdtree,
        marker="o",
        label="cKDTree",
    )

    plt.xlabel("Número de puntos $N$")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig(ruta_figura, dpi=300)
    plt.close()



def busqueda_exhaustiva(puntos, consultas, radio):
    """
    Realiza una búsqueda de vecinos dentro de un radio mediante búsqueda exhaustiva.

    Parámetros:
        puntos: Array de tamaño (N, 2) con los puntos almacenados.
        consultas: Array de tamaño (M, 2) con los puntos de consulta.
        radio: Radio de búsqueda.

    Devuelve:
        Lista o array con los índices de los puntos vecinos encontrados para cada consulta.
    """
    vecinos = []
    radio2 = radio ** 2
    for consulta in consultas:
        distancias = np.sum((puntos - consulta) ** 2, axis=1)
        vecinos.append(np.where(distancias <= radio2)[0])
    return vecinos
        



def busqueda_kdtree(puntos, consultas, radio):
    """
    Realiza una búsqueda de vecinos dentro de un radio utilizando scipy.spatial.cKDTree.

    Parámetros:
        puntos: Array de tamaño (N, 2) con los puntos almacenados.
        consultas: Array de tamaño (M, 2) con los puntos de consulta.
        radio: Radio de búsqueda.

    Devuelve:
        Lista con los índices de los puntos vecinos encontrados para cada consulta.
    """
    tree = cKDTree(puntos)
    vecinos = tree.query_ball_point(consultas, r=radio)
    return vecinos

#query_ball_point : If x is a single point, returns a list of the indices of the neighbors of x.
    #If x is an array of points, returns an object array of shape tuple containing lists of neighbors.
    #en nuestro caso, x=consultas, por lo que devuelve una lista de listas con los índices de los vecinos para cada consulta.
#consultas : The point or points to search for neighbors of.
#r:The radius of points to return, must broadcast to the length of x.


def medir_tiempos(N, dimension, radio, repeticiones, rng):
    """
    Mide los tiempos de ejecución de la búsqueda exhaustiva y con cKDTree para un tamaño N.

    Parámetros:
        N: Número de puntos almacenados y de consultas.
        dimension: Dimensión del espacio.
        radio: Radio de búsqueda.
        repeticiones: Número de repeticiones del experimento.
        rng: Generador aleatorio de NumPy.

    Devuelve:
        Diccionario con el tamaño N, el tiempo mediano del método exhaustivo,
        el tiempo mediano del método cKDTree y la aceleración obtenida.
    """
    tiempos_exhaustivo = []
    tiempos_kdtree = []

    #medir_exhaustivo = N <= 10000

    for _ in range(repeticiones):
        puntos, consultas = generar_datos(N, dimension, rng)

        #if medir_exhaustivo:
        t0 = time.perf_counter() # Inicia el temporizador para la búsqueda exhaustiva
        vecinos_exhaustivo = busqueda_exhaustiva(puntos, consultas, radio)
        t1 = time.perf_counter() # Detiene el temporizador para la búsqueda exhaustiva

        tiempos_exhaustivo.append(t1 - t0)
        #else:
        #    vecinos_exhaustivo = None

        t2 = time.perf_counter() # Inicia el temporizador para la búsqueda con cKDTree
        vecinos_kdtree = busqueda_kdtree(puntos, consultas, radio)
        t3 = time.perf_counter()

        tiempos_kdtree.append(t3 - t2)

        #if medir_exhaustivo:
        for v_exh, v_tree in zip(vecinos_exhaustivo, vecinos_kdtree):
            assert np.array_equal(np.sort(v_exh), np.sort(v_tree)) # Verifica que los vecinos encontrados por ambos métodos sean los mismos

    tiempo_exhaustivo = (
        np.median(tiempos_exhaustivo)# Calcula el tiempo mediano de la búsqueda exhaustiva
        #if medir_exhaustivo
        #else np.nan
    )

    tiempo_kdtree = np.median(tiempos_kdtree)

    aceleracion = (
        tiempo_exhaustivo / tiempo_kdtree
        #if medir_exhaustivo
        #else np.nan
    )

    return {
        "N": N,
        "tiempo_exhaustivo": tiempo_exhaustivo,
        "tiempo_kdtree": tiempo_kdtree,
        "aceleracion": aceleracion,
    }




def pendiente_loglog(N, tiempos):
    """
    Estima la pendiente empírica de una curva tiempo-tamaño en escala log-log.

    Parámetros:
        N: Array con los tamaños de entrada.
        tiempos: Array con los tiempos de ejecución asociados a cada tamaño.

    Devuelve:
        Pendiente de la recta ajustada a log(tiempos) frente a log(N).
    """
    N = np.asarray(N)
    tiempos = np.asarray(tiempos)
    mascara = np.isfinite(tiempos) & (tiempos > 0)

    coef = np.polyfit(np.log(N[mascara]), np.log(tiempos[mascara]), 1)
    return coef[0]


def main():
    """
    Ejecuta el benchmarking comparando búsqueda exhaustiva y búsqueda con cKDTree.

    El experimento genera puntos aleatorios en dimensión dos, mide los tiempos de
    búsqueda para distintos tamaños N y guarda los resultados en una tabla y una
    figura log-log.
    """
    resultados = []
    for N in tamanos:
        print(f"Midiendo tiempos para N={N}...")
        resultado = medir_tiempos(N, dimension, radio, repeticiones, rng)
        resultados.append(resultado)
        if np.isfinite(resultado["tiempo_exhaustivo"]):
            texto_exhaustivo = f"{resultado['tiempo_exhaustivo']:.6f}s"
            texto_aceleracion = f"{resultado['aceleracion']:.2f}x"
        else:
            texto_exhaustivo = "no medido"
            texto_aceleracion = "--"

        print(
            f"N={resultado['N']:6d} | "
            f"exhaustivo={texto_exhaustivo} | "
            f"cKDTree={resultado['tiempo_kdtree']:.6f}s | "
            f"aceleración={texto_aceleracion}"
        )

    guardar_resultados(resultados, "resultados/resultados_kdtree.csv")
    generar_figura_loglog(resultados, "resultados/benchmark_kdtree.png")

    N_array = np.array([res["N"] for res in resultados])
    t_exhaustivo = np.array([res["tiempo_exhaustivo"] for res in resultados])
    t_kdtree = np.array([res["tiempo_kdtree"] for res in resultados])

    pendiente_exhaustivo = pendiente_loglog(N_array, t_exhaustivo)
    pendiente_kdtree = pendiente_loglog(N_array, t_kdtree)

    print(f"Pendiente búsqueda exhaustiva: {pendiente_exhaustivo:.2f}")
    print(f"Pendiente cKDTree: {pendiente_kdtree:.2f}")
   


if __name__ == "__main__":
    main()
