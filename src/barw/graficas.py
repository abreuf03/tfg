import matplotlib.pyplot as plt
import numpy as np

# En este archivo se incluyen funciones para representar gráficamente
# los resultados de la simulación BARW.

def graficar_conducto(resultado, mostrar_puntas=True, guardar=None):
    """
    Dibuja la red de conductos generada por la simulación.

    Parámetros:
        resultado (dict): Diccionario devuelto por SimulacionBARW.ejecutar().
        mostrar_puntas (bool): Si es True, dibuja las puntas activas finales.
        guardar (str, opcional): Ruta donde guardar la figura. Si es None, no se guarda.
    """

    conducto = resultado["conducto"]
    puntas = resultado["puntas"]
    config = resultado["config"]

    plt.figure(figsize=(10, 5))

    for segmento in conducto:
        x0, y0, x1, y1 = segmento[:4]

        plt.plot([x0, x1], [y0, y1], color="black", linewidth=0.8)

    if mostrar_puntas:
        puntas_activas_x = [p.x for p in puntas if p.activa]
        puntas_activas_y = [p.y for p in puntas if p.activa]

        if len(puntas_activas_x) > 0:
            plt.scatter(puntas_activas_x, puntas_activas_y, color="red", s=20)

    plt.xlim(0, config.Lx)
    plt.ylim(0, config.Ly)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Red generada por el modelo BARW")
    plt.gca().set_aspect("equal", adjustable="box")

    if mostrar_puntas:
        plt.legend()

    plt.tight_layout()

    if guardar is not None:
        plt.savefig(guardar, dpi=300)

    plt.show()


def graficar_historial(resultado, guardar=None):
    """
    Dibuja la evolución temporal de la simulación.

    Parámetros:
        resultado (dict): Diccionario devuelto por SimulacionBARW.ejecutar().
        guardar (str, opcional): Ruta donde guardar la figura. Si es None, no se guarda.
    """

    historial = resultado["historial"]

    tiempo = historial["tiempo"]
    puntas_activas = historial["num_puntas_activas"]
    bifurcaciones = historial["num_bifurcaciones"]
    terminaciones = historial["num_terminaciones"]
    puntas_totales = historial["num_puntas_totales"]

    plt.figure(figsize=(9, 5))

    plt.plot(tiempo, puntas_activas, label="Puntas activas")
    plt.plot(tiempo, bifurcaciones, label="Bifurcaciones acumuladas")
    plt.plot(tiempo, terminaciones, label="Terminaciones acumuladas")
    plt.plot(tiempo, puntas_totales, label="Puntas totales")

    plt.xlabel("Tiempo")
    plt.ylabel("Número")
    plt.title("Evolución temporal de la simulación BARW")
    plt.legend()
    plt.tight_layout()

    if guardar is not None:
        plt.savefig(guardar, dpi=300)

    plt.show()


def graficar_xmax(resultado, guardar=None):
    """
    Dibuja la evolución temporal de x_max, la coordenada x máxima alcanzada por la red.

    Parámetros:
        resultado (dict): Diccionario devuelto por SimulacionBARW.ejecutar().
        guardar (str, opcional): Ruta donde guardar la figura. Si es None, no se guarda.
    """

    historial = resultado["historial"]

    tiempo = historial["tiempo"]
    x_max = historial["x_max"]

    plt.figure(figsize=(9, 5))

    plt.plot(tiempo, x_max, label="x_max")

    plt.xlabel("Tiempo")
    plt.ylabel("x_max")
    plt.title("Avance del frente de crecimiento (x_max) en la simulación BARW")
    plt.legend()
    plt.tight_layout()

    if guardar is not None:
        plt.savefig(guardar, dpi=300)

    plt.show()


def graficar_comparacion_semillas(df, guardar=None):
    """
    Representa el valor final de x_max para distintas semillas.

    Parámetros:
        df (pandas.DataFrame): Tabla con los resultados por semilla.
        guardar (str, opcional): Ruta donde guardar la figura.
    """

    plt.figure(figsize=(8, 5))

    plt.bar(
        df["semilla"].astype(str),
        df["x_max"]
    )

    plt.xlabel("Semilla")
    plt.ylabel(r"$x_{\max}$ final")
    plt.title("Comparación del avance máximo para distintas semillas")

    plt.tight_layout()

    if guardar is not None:
        plt.savefig(guardar, dpi=300)

    plt.show()


def graficar_resumen(resultado):
    """
    Muestra las dos gráficas principales de la simulación:
    la red generada y el historial temporal.
    """

    graficar_conducto(resultado)
    graficar_historial(resultado)

def graficar_boxplot_semillas(df, guardar=None):
    """
    Representa la variabilidad entre semillas para varias métricas.
    """

    metricas = [
        "x_max",
        "tiempo_final",
        "segmentos",
        "bifurcaciones",
        "terminaciones",
    ]

    for metrica in metricas:
        plt.figure(figsize=(6, 5))

        plt.boxplot(df[metrica], vert=True)
        plt.ylabel(metrica)
        plt.title(f"Distribución de {metrica} entre semillas")

        plt.tight_layout()

        if guardar is not None:
            nombre = guardar.replace(".png", f"_{metrica}.png")
            plt.savefig(nombre, dpi=300)

        plt.show()


def graficar_balance_bifurcacion_terminacion(df, guardar=None):
    """
    Representa el balance entre bifurcaciones y terminaciones.
    """

    plt.figure(figsize=(6, 5))

    plt.scatter(
        df["bifurcaciones"],
        df["terminaciones"],
        label="Simulaciones"
    )

    x_min = df["bifurcaciones"].min()
    x_max = df["bifurcaciones"].max()

    x = np.linspace(x_min, x_max, 100)
    y = x + 1

    plt.plot(x, y, linestyle="--", label=r"$N_{\mathrm{term}}=N_{\mathrm{bif}}+1$")

    plt.xlabel("Bifurcaciones")
    plt.ylabel("Terminaciones")
    plt.title("Balance entre bifurcación y terminación")
    plt.legend()
    plt.tight_layout()

    if guardar is not None:
        plt.savefig(guardar, dpi=300)

    plt.show()

def graficar_tiempo_busquedas(resumen, guardar=None):
    """
    Representa el tiempo medio de ejecución para cada método de búsqueda espacial.
    """

    metodos = resumen["metodo"]
    tiempos = resumen["tiempo_medio"]

    errores_inf = tiempos - resumen["ic95_inf"]
    errores_sup = resumen["ic95_sup"] - tiempos

    errores = [errores_inf, errores_sup]

    plt.figure(figsize=(7, 5))

    plt.bar(
        metodos,
        tiempos,
        yerr=errores,
        capsize=5
    )

    plt.xlabel("Método de búsqueda espacial")
    plt.ylabel("Tiempo medio de ejecución (s)")
    plt.title("Comparación del tiempo de ejecución")

    plt.tight_layout()

    if guardar is not None:
        plt.savefig(guardar, dpi=300)

    plt.show()


def graficar_tiempo_vs_segmentos(df, guardar=None):
    """
    Representa el tiempo de ejecución frente al número de segmentos generados.
    """

    plt.figure(figsize=(7, 5))

    for metodo, grupo in df.groupby("metodo"):
        plt.scatter(
            grupo["segmentos"],
            grupo["tiempo_ejecucion"],
            label=metodo,
            alpha=0.7
        )

    plt.xlabel("Número de segmentos generados")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("Tiempo de ejecución frente al tamaño de la red")
    plt.legend()

    plt.tight_layout()

    if guardar is not None:
        plt.savefig(guardar, dpi=300)

    plt.show()