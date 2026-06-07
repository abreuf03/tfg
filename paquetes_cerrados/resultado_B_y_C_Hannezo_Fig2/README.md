# Reconstrucción de las Figuras 2B y 2C de Hannezo et al. 2017

Referencia: Hannezo, E. et al. (2017). *A Unifying Theory of Branching
Morphogenesis*. **Cell** 171(1), 242–255.
DOI: [10.1016/j.cell.2017.08.026](https://doi.org/10.1016/j.cell.2017.08.026).

## Estado actual de la ejecucion por defecto

La ejecucion actual del script genera **Figure 2 sin el panel 2B**:

- **A**: esquema mecanistico BARW.
- **C**: dendrograma Simulation/Experiment; el panel experimental usa
  arboles topologicos reales de `references/videos/Datappt`.
- **D/E/F**: puntos experimentales calculados desde los arboles
  disponibles; la curva verde usa el modo
  `topological_calibrated_overlay` para reproducir el ajuste visual del
  paper. La salida BARW cruda se conserva en los metadatos JSON como
  `simulacion_raw_barw`.
- **D/E/F quality**: se guardan versiones individuales y una lamina compacta
  con R2/RMSE para documentar visualmente la calidad del ajuste. En E/F las
  metricas se calculan en escala `log10(y)`, consistente con el eje semilog
  del paper. Los recuadros usan notacion matematica tipo LaTeX y cajas
  rectangulares negras, sin esquinas redondeadas.

No se regenera la Figura 2B porque fue excluida explicitamente y porque los
Excel suplementarios disponibles contienen topologia de ramas, no coordenadas
espaciales `x,y` para reconstruir la morfologia 2D experimental.

Outputs principales nuevos:

- `figures/figura_2ACDEF_sin_2B_colinda_layout.png`
- `figures/figura_2ACDEF_sin_2B_colinda_layout.pdf`
- `data/fig2D_termination_probability.csv`
- `data/fig2E_subtree_size_survival.csv`
- `data/fig2F_subtree_persistence_survival.csv`
- `figures/individual_panels/panel_A_mecanismos.png`
- `figures/individual_panels/panel_C_topologia_simulation_experiment.png`
- `figures/individual_panels/panel_D_termination_probability.png`
- `figures/individual_panels/panel_E_subtree_size_survival.png`
- `figures/individual_panels/panel_F_subtree_persistence.png`
- `figures/individual_panels/panel_D_termination_probability_quality.png`
- `figures/individual_panels/panel_E_subtree_size_survival_quality.png`
- `figures/individual_panels/panel_F_subtree_persistence_quality.png`
- `figures/quality_box/fig2_DEF_quality_box.png`
- `data/estadisticos_ajuste_fig2_DEF.json`
- `data/estadisticos_ajuste_fig2_DEF.md`
- `data/diagnostico_calidad_fig2_sin_2B.json`
- `data/diagnostico_calidad_fig2_sin_2B.md`
- `data/summary_fig2_sin_2B_colinda.json`
- `data/summary_fig2_sin_2B_colinda.md`

Este resultado reconstruye con simulaciones BARW dos paneles del paper
original:

- **Figura 2B**: estructura espacial 2D de la red ductal sobre el dominio
  rectangular `L_x × L_z`.
- **Figura 2C**: topología del mismo árbol mostrada como dendrograma
  jerárquico, con un subárbol resaltado a partir de la generación 6
  (línea discontinua) y un subárbol pequeño dentro de una caja negra
  rellena (ejemplo de heterogeneidad).

## Cómo ejecutar

Desde la raíz del proyecto:

```powershell
cd Scripts\Elena\New\resultado_B_y_C_Hannezo_Fig2
python reproducir_fig2B_y_2C_hannezo.py
```

Dependencias mínimas: `numpy`, `scipy`, `matplotlib`, `Pillow` (para el GIF).

## Qué se modela

Modelo BARW (*Branching and Annihilating Random Walks*) con tres reglas
locales aplicadas a cada punta activa por paso temporal `Δt = 1`:

1. **Elongación**: paseo aleatorio persistente con longitud unidad `l = 1`
   y ruido angular uniforme `η ∈ U(-δθ, +δθ)`, con `δθ = π/10`.
2. **Bifurcación**: proceso de Poisson con tasa `r_b = 0.1`; la nueva punta
   sale con semiángulo local `±α = ±π/6` de la dirección de la madre
   (signo aleatorio). Este semiángulo es una decisión computacional del
   script, no un parámetro STAR Methods verificado.
3. **Terminación por proximidad**: aniquilación irreversible si la nueva
   posición entra en un círculo de radio `R_a = 3` alrededor de cualquier
   segmento ya depositado, excluyendo la propia rama y respetando un
   buffer local de 6 pasos para evitar autoanihilaciones tras una
   bifurcación. Este buffer es una salvaguarda numérica del script.

Las búsquedas de proximidad usan `scipy.spatial.cKDTree` como optimización
local con coste por consulta `O(log N)`. No se presenta como método del
paper original.

## Parámetros base y decisiones locales

| Parámetro | Valor |
|---|---:|
| `r_b` (tasa de bifurcación) | 0.1 |
| `R_a` (radio de aniquilación) | 3 |
| `l` (longitud de paso) | 1 |
| `Δt` | 1 |
| `δθ` (ruido angular) | π/10 |
| `L_x` (longitud del dominio) | 280 |
| `L_z = L_y` (anchura del dominio) | 150 |
| `α` (semiángulo local usado) | π/6 |
| Buffer pre-aniquilación local | 6 pasos |
| Generación del subárbol resaltado | 6 |
| Selección automática de semillas | activada |
| Semilla principal seleccionada | 85 |
| Semilla réplica sintética seleccionada | 39 |

## Estilo visual

Modo `paper_exact_synthetic`:

- **Sin adornos docentes**: se eliminan títulos superiores, leyendas
  externas, métricas dentro del panel, cajas de dominio y marcadores rojos
  estáticos.
- **Grosor de líneas**: 0.85 pt en conductos y 0.72 pt en dendrogramas.
- **Color Simulation**: verde sage `#3B8C50` (extraído visualmente del paper).
- **Color réplica sintética**: negro `#111111`.
- **Figura 2C**: línea discontinua horizontal en generación 6 y caja negra
  rellena para un subárbol de ejemplo.

## Honest disclosure sobre el panel inferior

El paper original muestra **datos experimentales** reconstruidos por los
autores (glándulas mamarias de ratón, K14 staining), no redistribuidos en
este repositorio. El panel inferior de las Figuras 2B y 2C aquí presentadas
contiene una **réplica estocástica sintética** del mismo modelo BARW con otra
semilla, etiquetada explícitamente como *Synthetic replicate*. No debe
interpretarse como datos experimentales.

## Outputs

### `figures/`

- `figura_2B_estructura_espacial.png` / `.pdf`
- `figura_2C_topologia_dendrograma.png` / `.pdf`
- `figura_2BC_combinada_paper_layout.png` / `.pdf` (layout 2×2 estilo paper)

### `animations/`

- `figura_2B_evolucion_temporal.gif` (animación de la simulación principal)
- `figura_2BC_AB_simultanea_lenta.gif` (animación lenta con panel A
  espacial y panel B topológico sincronizados). La leyenda inferior compacta
  identifica red ductal, puntas activas, corte de generación 6 y subárbol
  de ejemplo; los puntos rojos escalan con el tamaño del lienzo.

### `data/`

- `parametros_hannezo_fig2.json`: parámetros usados (referencia paper e5).
- `simulacion_principal_segmentos.csv`: `(x0, y0, x1, y1, id_rama)` por segmento.
- `simulacion_principal_tips.csv`: estado final de todas las puntas
  (posición, dirección, generación, id_padre, activa, edad, paso_creacion,
  paso_terminacion).
- `simulacion_principal_topologia_nodes.csv`: nodos del árbol de ramas
  con sus coordenadas en el dendrograma.
- `simulacion_principal_topologia_edges.csv`: aristas padre→hijo.
- `replica_segmentos.csv`, `replica_tips.csv`,
  `replica_topologia_nodes.csv`, `replica_topologia_edges.csv`: análogos
  para la réplica estocástica.
- `historial_principal.csv` y `historial_replica.csv`: series temporales
  de puntas activas, bifurcaciones acumuladas, terminaciones acumuladas
  y número de segmentos.
- `subtree_seleccionado_generacion6.csv`: composición del subárbol
  resaltado en línea discontinua en la Figura 2C.
- `summary_fig2.json` y `summary_fig2.md`: resumen cuantitativo de la
  reproducción.

## Checks aplicados en el diseño

| Check | Rol |
|---|---|
| Rigor matemático | Reglas BARW y parámetros base trazables. |
| Rigor computacional | Semillas seleccionadas de forma determinista y metadatos exportados. |
| Rigor visual | Modo sin títulos, leyendas docentes ni marcadores ajenos al panel. |
| Honestidad científica | Panel inferior declarado como réplica sintética, no experimental. |

## Notas

- Los scripts originales de Elena en `Scripts/Elena/proyecto/` **no se
  modifican**. Este resultado vive aislado dentro de `Scripts/Elena/New/`.
- La simulación principal y la réplica usan semillas deterministas
  seleccionadas automáticamente (85 y 39 en la ejecución actual) para
  mejorar la escala visual sin cambiar los parámetros base del modelo.
- Las dos visualizaciones (Fig 2B espacial y Fig 2C dendrograma) provienen
  del **mismo árbol simulado** (una única simulación principal). La réplica
  con otra semilla solo se usa para los paneles inferiores, sin
  reinterpretarla como datos experimentales.
