from src.barw.config import BARWConfig
from src.barw.simulacion import SimulacionBARW
from src.barw.graficas import graficar_conducto, graficar_historial, graficar_xmax


def main():
    config = BARWConfig()

    kdtree = True # Cambia a False para usar búsqueda exhaustiva

    simulacion = SimulacionBARW(config=config, usar_kdtree=kdtree)
    resultado = simulacion.ejecutar()



    if(kdtree):
        graficar_conducto(resultado, guardar="resultados/barw_conducto_kdtree.png")
        graficar_historial(resultado, guardar="resultados/barw_historial_kdtree.png")
        #graficar_xmax(resultado, guardar="resultados/barw_xmax_kdtree.png")
        #graficar_conducto(resultado, guardar="resultados/barw_conducto_50g.png")
        #graficar_historial(resultado, guardar="resultados/barw_historial_50g.png")
    
    else:
        graficar_conducto(resultado, guardar="resultados/barw_conducto_exhaustiva.png")
        graficar_historial(resultado, guardar="resultados/barw_historial_exhaustiva.png")
        graficar_xmax(resultado, guardar="resultados/barw_xmax_exhaustiva.png")
    #semillas = [1, 2, 3, 4, 5, 10, 42, 100]

    #for semilla in semillas:
    #    config = BARWConfig()
    #    config.semilla = semilla

    #    simulacion = SimulacionBARW(config=config, usar_kdtree=True)
    #    resultado = simulacion.ejecutar()

    #    historial = resultado["historial"]
    #    conducto = resultado["conducto"]

    #    x_max = max(max(seg[0], seg[2]) for seg in conducto) if conducto else 0

    #    print(f"Semilla: {semilla}")
    #    print(f"  Segmentos: {len(conducto)}")
    #    print(f"  Bifurcaciones: {historial['num_bifurcaciones'][-1]}")
    #    print(f"  Terminaciones: {historial['num_terminaciones'][-1]}")
    #    print(f"  Puntas finales: {historial['num_puntas_activas'][-1]}")
    #    print(f"  x_max: {x_max:.2f}")
    #    print()
    


if __name__ == "__main__":
    main()