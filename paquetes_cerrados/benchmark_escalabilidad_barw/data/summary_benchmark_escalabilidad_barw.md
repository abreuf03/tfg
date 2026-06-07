# Benchmark de escalabilidad BARW

## Objetivo

Evaluar la regla computacional mas costosa del simulador BARW: la busqueda de conductos cercanos dentro del radio de aniquilacion. El benchmark usa una geometria BARW reproducible y compara busqueda exhaustiva, cKDTree, grid espacial y quadtree.

## Carga BARW generada

- Puntos de conducto: 50000
- Segmentos: 49999
- Consultas registradas: 11213
- Ramas generadas: 7975
- Terminaciones: 7565
- Raices: 1
- Pasos simulados: 539

## Resultados principales

- Mayor tamano benchmarkeado: N = 32000
- Busqueda exhaustiva medida hasta: N = 8000
- Equivalencia exacta de vecinos: True
- Calidad operativa estimada: 10.0/10

## Pendientes log-log empiricas

- Exhaustiva: 1.273
- Grid espacial: 0.970
- cKDTree: 0.827
- Quadtree: 0.839

## Speedup en el limite con referencia exhaustiva

- Grid espacial: 0.55x
- cKDTree: 4.29x
- Quadtree: 0.51x

## Interpretacion para la memoria

Este bloque no reproduce una figura concreta de Hannezo et al. (2017). Aporta la parte informatica del TFG: demuestra que la regla de terminacion por proximidad puede acelerarse sin modificar los vecinos detectados ni la regla biologica simulada.

Figuras recomendadas para la memoria:

- Panel A: geometria BARW usada como carga de consultas.
- Panel C: escalabilidad temporal en escala log-log.
- Panel D: aceleracion respecto a busqueda exhaustiva.
- Panel B: verificacion de equivalencia exacta de vecinos.

Para la defensa conviene usar la figura compuesta completa y explicar que el color identifica el metodo, mientras que la regla fisica de aniquilacion permanece fija.

## Archivos

- `figures/benchmark_escalabilidad_barw_layout.png`
- `figures/benchmark_escalabilidad_barw_layout.pdf`
- `figures/individual_panels/*.png`
- `data/benchmark_scaling.csv`
- `data/benchmark_metrics.json`
