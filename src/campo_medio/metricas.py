
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from src.campo_medio.config import CampoMedioConfig
from src.malla import Malla


# -----------------------------------------------------------------------------
# Métricas generales para la validación numérica del Capítulo 2
# -----------------------------------------------------------------------------


def norma_l2_discreta(valores: np.ndarray, dx: float) -> float:
    """
        Calcula la norma L2 discreta de un conjunto de valores.

        La norma se aproxima mediante la expresión:

            sqrt(dx * sum_j |valores_j|^2)

        Parámetros:
            valores: Array con los valores de la función en los nodos de la malla espacial.
            dx: Paso de discretización espacial.

        Devuelve:
            El valor de la norma L2 discreta de los valores proporcionados.
    """
    valores = np.asarray(valores, dtype=float)
    return float(np.sqrt(dx * np.sum(valores**2)))


def resumen_funcional(
    A: np.ndarray,
    I: np.ndarray,
    malla: Malla,
    config: CampoMedioConfig,
    tolerancia: float = 1.0e-12,
) -> dict[str, float | bool]:
    """
    Comprueba las propiedades estructurales básicas de una simulación.

    Parámetros:
        A: Matriz con los valores de la variable activa.
        I: Matriz con los valores de la variable inactiva.
        malla: Malla espacio-temporal empleada en la simulación.
        config: Configuración del modelo y de las condiciones de contorno.
        tolerancia: Tolerancia numérica utilizada en las comprobaciones.

    Devuelve:
        Un diccionario con los resultados de las comprobaciones funcionales,
        incluyendo las dimensiones, la finitud de la solución, las condiciones
        de contorno, la no negatividad y el carácter no decreciente de I.
    """
    forma_esperada = (malla.nt, malla.nx)
    incremento_minimo_i = float(np.min(np.diff(I, axis=0)))
    minimo_a = float(np.min(A))
    minimo_i = float(np.min(I))

    return {
        "forma_correcta": A.shape == forma_esperada and I.shape == forma_esperada,
        "solucion_finita": bool(np.isfinite(A).all() and np.isfinite(I).all()),
        "frontera_izquierda": bool(
            np.allclose(A[:, 0], config.a_in, atol=tolerancia, rtol=0.0)
        ),
        "frontera_derecha": bool(
            np.allclose(A[:, -1], config.a_out, atol=tolerancia, rtol=0.0)
        ),
        "min_A": minimo_a,
        "max_A": float(np.max(A)),
        "min_I": minimo_i,
        "max_I": float(np.max(I)),
        "no_negatividad": minimo_a >= -tolerancia and minimo_i >= -tolerancia,
        "incremento_minimo_I": incremento_minimo_i,
        "I_no_decreciente": incremento_minimo_i >= -tolerancia,
    }


def comparar_soluciones_finales(
    A_1: np.ndarray,
    I_1: np.ndarray,
    A_2: np.ndarray,
    I_2: np.ndarray,
    dx: float,
) -> dict[str, float]:
    
    """
    Compara dos soluciones numéricas en el instante temporal final.

    La comparación se realiza mediante los errores absolutos en norma infinito
    y norma L2 discreta para las variables activa e inactiva. También se
    calculan los errores relativos en norma infinito, tomando la segunda
    solución como referencia.

    Parámetros:
        A_1: Matriz con los valores de la variable activa de la primera solución.
        I_1: Matriz con los valores de la variable inactiva de la primera solución.
        A_2: Matriz con los valores de la variable activa de la solución de referencia.
        I_2: Matriz con los valores de la variable inactiva de la solución de referencia.
        dx: Paso de discretización espacial utilizado para calcular la norma L2.

    Devuelve:
        Un diccionario con los errores absolutos en norma infinito y norma L2
        discreta, junto con los errores relativos en norma infinito para las
        variables activa e inactiva.
    """
    diferencia_a = np.asarray(A_1[-1] - A_2[-1], dtype=float)
    diferencia_i = np.asarray(I_1[-1] - I_2[-1], dtype=float)

    max_a_ref = max(float(np.max(np.abs(A_2[-1]))), np.finfo(float).eps)
    max_i_ref = max(float(np.max(np.abs(I_2[-1]))), np.finfo(float).eps)

    error_a_inf = float(np.max(np.abs(diferencia_a)))
    error_i_inf = float(np.max(np.abs(diferencia_i)))

    return {
        "error_A_inf": error_a_inf,
        "error_A_l2": norma_l2_discreta(diferencia_a, dx),
        "error_I_inf": error_i_inf,
        "error_I_l2": norma_l2_discreta(diferencia_i, dx),
        "error_rel_A_inf": error_a_inf / max_a_ref,
        "error_rel_I_inf": error_i_inf / max_i_ref,
    }


def error_entre_mallas(
    x_gruesa: np.ndarray,
    u_gruesa: np.ndarray,
    x_fina: np.ndarray,
    u_fina: np.ndarray,
) -> dict[str, float]:
    """
    Calcula el error entre dos perfiles obtenidos con mallas espaciales distintas.

    La solución calculada sobre la malla fina se interpola linealmente en los
    nodos de la malla gruesa. A continuación, se compara con la solución de la
    malla gruesa mediante la norma infinito y la norma L2 discreta.

    Parámetros:
        x_gruesa: Array con los nodos espaciales de la malla gruesa.
        u_gruesa: Array con los valores del perfil calculado sobre la malla gruesa.
        x_fina: Array con los nodos espaciales de la malla fina.
        u_fina: Array con los valores del perfil calculado sobre la malla fina.

    Devuelve:
        Un diccionario con el error en norma infinito y el error en norma L2
        discreta entre ambos perfiles.
    """
    x_gruesa = np.asarray(x_gruesa, dtype=float)
    u_gruesa = np.asarray(u_gruesa, dtype=float)
    x_fina = np.asarray(x_fina, dtype=float)
    u_fina = np.asarray(u_fina, dtype=float)

    referencia = np.interp(x_gruesa, x_fina, u_fina)
    diferencia = u_gruesa - referencia
    dx = float(x_gruesa[1] - x_gruesa[0])

    return {
        "inf": float(np.max(np.abs(diferencia))),
        "l2": norma_l2_discreta(diferencia, dx),
    }


def orden_observado(
    error_h: float,
    error_h2: float,
    razon_refinamiento: float = 2.0,
) -> float:
    """
    Calcula el orden de convergencia observado entre dos niveles de refinamiento.

    El orden se obtiene mediante la expresión:

        p = log(error_h / error_h2) / log(razon_refinamiento)

    Parámetros:
        error_h: Error obtenido con la malla de paso h.
        error_h2: Error obtenido con la malla refinada.
        razon_refinamiento: Cociente entre los pasos de las dos mallas.
            Por defecto, se considera un refinamiento de razón 2.

    Devuelve:
        El orden de convergencia observado. Si alguno de los errores no es
        estrictamente positivo, devuelve NaN.
    """
    if error_h <= 0.0 or error_h2 <= 0.0:
        return float("nan")
    return float(np.log(error_h / error_h2) / np.log(razon_refinamiento))


def ultimo_punto_significativo(
    x: np.ndarray,
    perfil: np.ndarray,
    tolerancia: float = 1.0e-4,
) -> float:
    """
    Determina la última posición en la que un perfil supera una tolerancia.

    La búsqueda se realiza sobre el valor absoluto del perfil. Si ningún valor
    supera la tolerancia indicada, se devuelve la primera posición del dominio.

    Parámetros:
        x: Array con las posiciones espaciales.
        perfil: Array con los valores del perfil en cada posición.
        tolerancia: Umbral absoluto utilizado para considerar que el perfil
            es significativo.

    Devuelve:
        La última posición espacial en la que el valor absoluto del perfil
        supera la tolerancia. Si no existe ninguna, devuelve x[0].
    """
    x = np.asarray(x, dtype=float)
    perfil = np.asarray(perfil, dtype=float)
    indices = np.flatnonzero(np.abs(perfil) > tolerancia)
    return float(x[indices[-1]]) if indices.size else float(x[0])


def serie_residuo_balance(
    A: np.ndarray,
    I: np.ndarray,
    malla: Malla,
    config: CampoMedioConfig,
    metodo: Literal["euler", "imex"],
) -> np.ndarray:
    """
    Calcula el residuo máximo de la identidad discreta de balance en cada paso.

    Al sumar las ecuaciones continuas de las variables activa e inactiva se
    obtiene la identidad:

        (a + i)_t = D * a_xx + (rb + re) * a

    Para el método de Euler, el término de difusión se evalúa en el instante
    t^n. Para el método IMEX--Crank--Nicolson, se utiliza el promedio de los
    laplacianos evaluados en los instantes t^n y t^{n+1}.

    Parámetros:
        A: Matriz con los valores de la variable activa. Cada fila corresponde
            a un instante temporal y cada columna a un nodo espacial.
        I: Matriz con los valores de la variable inactiva. Cada fila corresponde
            a un instante temporal y cada columna a un nodo espacial.
        malla: Malla espacio-temporal empleada en la simulación.
        config: Configuración del modelo, incluyendo los parámetros D, rb y re.
        metodo: Método numérico utilizado. Debe ser "euler" o "imex".

    Devuelve:
        Un array con el residuo máximo absoluto de la identidad discreta de
        balance para cada paso temporal.

    Lanza:
        ValueError: Si el método indicado no es "euler" ni "imex".
    """
    A = np.asarray(A, dtype=float)
    I = np.asarray(I, dtype=float)

    lhs = (
        (A[1:, 1:-1] - A[:-1, 1:-1])
        + (I[1:, 1:-1] - I[:-1, 1:-1])
    ) / malla.dt

    lap_n = (
        A[:-1, :-2] - 2.0 * A[:-1, 1:-1] + A[:-1, 2:]
    ) / malla.dx**2

    if metodo == "euler":
        difusion = config.D * lap_n
    elif metodo == "imex":
        lap_np1 = (
            A[1:, :-2] - 2.0 * A[1:, 1:-1] + A[1:, 2:]
        ) / malla.dx**2
        difusion = 0.5 * config.D * (lap_n + lap_np1)
    else:
        raise ValueError("metodo debe ser 'euler' o 'imex'.")

    rhs = difusion + (config.rb + config.re) * A[:-1, 1:-1]
    return np.max(np.abs(lhs - rhs), axis=1)


def residuo_balance_discreto(
    A: np.ndarray,
    I: np.ndarray,
    malla: Malla,
    config: CampoMedioConfig,
    metodo: Literal["euler", "imex"],
) -> float:
    """
    Calcula el máximo global del residuo de la identidad discreta de balance.

    El residuo se obtiene para cada paso temporal mediante la función
    serie_residuo_balance. A continuación, se devuelve el mayor valor absoluto
    alcanzado durante toda la simulación.

    Parámetros:
        A: Matriz con los valores de la variable activa.
        I: Matriz con los valores de la variable inactiva.
        malla: Malla espacio-temporal empleada en la simulación.
        config: Configuración del modelo, incluyendo sus parámetros.
        metodo: Método numérico utilizado. Debe ser "euler" o "imex".

    Devuelve:
        El máximo global del residuo de balance discreto.
    """
    return float(np.max(serie_residuo_balance(A, I, malla, config, metodo)))

# cap 4


def posicion_maximo_parabolico(
    x: np.ndarray,
    perfil: np.ndarray,
) -> float:
    """
    Estima la posición del máximo de un perfil mediante interpolación parabólica.

    Primero se localiza el nodo en el que el perfil alcanza su valor máximo.
    Si dicho nodo es interior, se utiliza el valor del perfil en ese nodo y en
    sus dos vecinos para ajustar localmente una parábola y estimar la posición
    del máximo con precisión subnodal.

    Parámetros:
        x: Array unidimensional con las posiciones espaciales.
        perfil: Array unidimensional con los valores del perfil en cada nodo.

    Devuelve:
        La posición estimada del máximo del perfil. Si el máximo se encuentra
        en una frontera o la interpolación parabólica no puede realizarse, se
        devuelve la posición del nodo de máximo valor.

    Lanza:
        ValueError: Si x o perfil no son vectores unidimensionales, no tienen
            la misma longitud o contienen menos de tres nodos.
    """
    x = np.asarray(x, dtype=float)
    perfil = np.asarray(perfil, dtype=float)

    if x.ndim != 1 or perfil.ndim != 1:
        raise ValueError("x y perfil deben ser vectores unidimensionales.")

    if x.size != perfil.size:
        raise ValueError("x y perfil deben tener la misma longitud.")

    if x.size < 3:
        raise ValueError("Se necesitan al menos tres nodos.")

    indice_maximo = int(np.argmax(perfil))

    if indice_maximo == 0 or indice_maximo == perfil.size - 1:
        return float(x[indice_maximo])

    y_izquierda = perfil[indice_maximo - 1]
    y_centro = perfil[indice_maximo]
    y_derecha = perfil[indice_maximo + 1]

    denominador = y_izquierda - 2.0 * y_centro + y_derecha

    if np.isclose(denominador, 0.0):
        return float(x[indice_maximo])

    desplazamiento = (
        0.5
        * (y_izquierda - y_derecha)
        / denominador
    )

    desplazamiento = float(
        np.clip(desplazamiento, -1.0, 1.0)
    )

    dx = float(
        x[indice_maximo + 1] - x[indice_maximo]
    )

    return float(
        x[indice_maximo] + desplazamiento * dx
    )


def trayectoria_pico(
    x: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """
    Calcula la posición del máximo de A en cada instante.

    Parámetros:
        x: Array unidimensional con las posiciones espaciales.
        A: Matriz con los valores de la variable activa. Cada fila corresponde
            a un instante temporal y cada columna a un nodo espacial.

    Devuelve:
        Un array con las posiciones estimadas del máximo de A para cada instante.
    """
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)

    if x.ndim != 1:
        raise ValueError("x debe ser un vector unidimensional.")

    if A.ndim != 2:
        raise ValueError("A debe ser una matriz de tamaño nt x nx.")

    if A.shape[1] != x.size:
        raise ValueError(
            "El número de columnas de A debe coincidir con la longitud de x."
        )

    posiciones = np.empty(A.shape[0], dtype=float)

    for n in range(A.shape[0]):
        posiciones[n] = posicion_maximo_parabolico(
            x,
            A[n],
        )

    return posiciones


