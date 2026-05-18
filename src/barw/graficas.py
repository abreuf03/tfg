import matplotlib.pyplot as plt


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


def graficar_resumen(resultado):
    """
    Muestra las dos gráficas principales de la simulación:
    la red generada y el historial temporal.
    """

    graficar_conducto(resultado)
    graficar_historial(resultado)