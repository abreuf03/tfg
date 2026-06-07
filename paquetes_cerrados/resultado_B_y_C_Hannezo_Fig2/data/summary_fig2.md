# Resumen Figura 2B y 2C  -  Hannezo et al. 2017

Generado: 2026-05-26T13:12:02
Python:   3.13.13
Plataforma: Windows-11-10.0.26200-SP0

## Parámetros del paper (STAR-Methods e5, mammary default)

El panel inferior es una réplica sintética porque los datos
experimentales reconstruidos del paper no están disponibles en
este repositorio.

| Parámetro | Valor |
|---|---:|
| r_b | 0.1 |
| R_a | 3 |
| l | 1 |
| delta_t | 1 |
| delta_theta | 0.314159 |
| L_x | 280 |
| L_y | 150 |
| alpha_bifurcacion | 0.523599 |
| pasos_exclusion_aniquilacion | 6 |
| max_pasos | 1200 |
| max_puntas | 50000 |
| subtree_generacion_resaltar | 6 |
| semilla_principal | 85 |
| semilla_replica | 39 |
| auto_seleccionar_semillas | True |
| seed_search_min | 0 |
| seed_search_max | 250 |
| seed_search_target_ramas | 300 |
| seed_search_target_generacion | 45 |
| seed_search_target_xmax | 260 |

## Resultados cuantitativos

| Métrica | Simulación principal | Réplica estocástica |
|---|---:|---:|
| Semilla | 85 | 39 |
| N segmentos | 2711 | 2802 |
| N puntas totales | 292 | 290 |
| N bifurcaciones | 291 | 289 |
| N terminaciones | 292 | 290 |
| Balance q (term/(term+bif)) | 0.5009 | 0.5009 |
| Generación máxima | 54 | 63 |
| N ramas en árbol | 292 | 290 |

## Modelo (BARW, Hannezo et al. 2017)

Tres reglas locales aplicadas a cada punta activa por paso temporal:

1. **Elongación**: paseo aleatorio persistente con longitud unidad
   `l` y ruido angular uniforme en `[-delta_theta, delta_theta]`.
2. **Bifurcación**: proceso de Poisson con tasa `r_b`; el ángulo
   relativo a la dirección de la madre es `+/- alpha_bifurcacion`.
3. **Terminación**: aniquilación irreversible si la nueva posición
   está dentro del radio `R_a` de cualquier punto de la red ya
   depositada, excluyendo la propia rama y respetando el buffer
   `pasos_exclusion_aniquilacion`.

Las búsquedas de proximidad emplean `scipy.spatial.cKDTree`, con
coste por consulta O(log N) (Bentley, 1975).

## Honest disclosure sobre el panel inferior

El paper original muestra reconstrucciones experimentales de
glándulas mamarias de ratón (no redistribuidas en este repositorio).
Aquí, el panel inferior contiene una réplica estocástica sintética
del mismo modelo con otra semilla, etiquetada explícitamente como
*Synthetic replicate*. No debe interpretarse como datos
experimentales.

## Archivos generados

- `figures/figura_2B_estructura_espacial.{png,pdf}`
- `figures/figura_2C_topologia_dendrograma.{png,pdf}`
- `figures/figura_2BC_combinada_paper_layout.{png,pdf}`
- `animations/figura_2B_evolucion_temporal.gif`
- `animations/figura_2BC_AB_simultanea_lenta.gif`
- `data/parametros_hannezo_fig2.json`
- `data/simulacion_principal_segmentos.csv`
- `data/simulacion_principal_tips.csv`
- `data/simulacion_principal_topologia_nodes.csv`
- `data/simulacion_principal_topologia_edges.csv`
- `data/replica_segmentos.csv`
- `data/replica_tips.csv`
- `data/replica_topologia_nodes.csv`
- `data/replica_topologia_edges.csv`
- `data/subtree_seleccionado_generacion6.csv`
- `data/historial_principal.csv`
- `data/historial_replica.csv`
- `data/summary_fig2.{json,md}`
