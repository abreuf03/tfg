from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW
from src.barw.graficas import graficar_conducto, graficar_historial


def main():
    config = BARWConfig()

    simulacion = SimulacionBARW(config=config, usar_kdtree=False)
    resultado = simulacion.ejecutar()

    graficar_conducto(resultado)
    graficar_historial(resultado)


if __name__ == "__main__":
    main()