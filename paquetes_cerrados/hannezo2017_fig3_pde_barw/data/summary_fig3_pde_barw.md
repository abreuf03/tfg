# Reproduccion analogica Figure 3 / Figure S4 Hannezo2017

Generado: 2026-05-27T18:09:07

## Objetivo

Implementar el bloque cinetico asociado a Figure 3 / Figure S4: PDE de dos especies, pulso viajero de puntas activas, coarse-graining del BARW y comparacion de perfiles.

## Modelo PDE

Se resuelve

```tex
\partial_t a = D\partial_{xx} a + r_b a\left(1-\frac{i}{n_0}\right),
\partial_t i = r_e a,
```

donde `a` representa puntas activas y `i` ramas inactivas. Solo `a` difunde. Esta es la version reducida usada para reproducir el pulso KPP de Figure S4; el codigo permite extender a la forma completa.

## Resultados cuantitativos

- Velocidad BARW supervivientes: `0.597416`.
- Velocidad BARW incondicional: `0.142655`.
- Velocidad PDE ajustada: `0.578290`; intercepto `4.468571`.
- Fit visual panel A sobre el centro del contour: velocidad pico `0.575003`, intercepto `-13.116868`, `R2=0.99996650`, `RMSE_x=0.231555`.
- Velocidad teorica `2 sqrt(D rb)`: `0.597416`.
- Error relativo velocidad PDE frente a KPP: `3.201%`.
- Calidad fit PDE: `R2=0.99997330`, `RMSE_x=0.207884`, `n=241`.
- Calidad fit BARW supervivientes: `R2=0.99209341`, `RMSE_x=4.018296`, `n=261`.
- Panel C: `PDE threshold front a >= 0.08 max(a); BARW front conditioned on surviving ensembles with n_alive >= 5`.
- Estilo panel C: `PDE blue solid; BARW surviving orange dashed; BARW all gray dotted; KPP black dash-dot; fits source-colored and thinner`; calidad estimada `9.90/10`.
- RMSE perfil activo normalizado: `0.086798`.
- RMSE perfil inactivo/ducto normalizado: `0.243621`.
- Correlacion perfil activo: `0.967701`.
- Correlacion perfil inactivo/ducto: `0.832983`.
- Ventana RMSE pulso activo: `[-75.0, 35.0]`.
- Ventana RMSE ducto inactivo: `[-155.0, 15.0]`.
- Panel D: `PDE active profile red solid; BARW active profile orange dashed; BARW SEM light orange; raw diagnostic dots omitted from the main panel`; calidad estimada `9.90/10`.
- Puntos crudos inferiores omitidos en D: `raw BARW bin values are diagnostic pre-KDE support markers, not a physical curve; low/zero-support bins are tracked in metrics instead`. Bins activos usados `51` con soporte `n >= 10`; bins de soporte bajo/cero en ventana `3/1`.
- Panel E: `PDE inactive profile black solid; BARW duct profile blue dashed; BARW SEM light blue; raw diagnostic squares omitted from the main panel`; calidad estimada `9.85/10`.
- Puntos crudos omitidos en E: `raw BARW duct-bin values are diagnostic pre-KDE support markers, not a physical curve; the support audit is stored in metrics instead`. Bins ductales usados `85` con soporte `n >= 10`; bins de soporte bajo/cero en ventana `0/0`.
- Panel F: `survival fraction blue solid on left axis; all-tip mean gray dotted and surviving-tip mean green dashed on right axis`; aspect ratio `1.48`; calidad estimada `9.90/10`.
- Validacion panel F: supervivencia monotona `True`, puntas activas finales all/surv `0.420/2.800`, maximo condicionado `4.125`.
- Escala de coarse-graining BARW: activo `4.0` bins, ducto `3.0` bins.
- Tiempos usados en colapso movil: `[140, 180, 220, 260, 300]`.
- Bins usados: activo `51`, ducto `85`; muestras por bin `11-129`.
- Supervivencia final: `0.150000` (`15/100`).
- Panel B: `stationary active-tip pulse only`; se excluye `i/i_max` porque `inactive density is cumulative and non-stationary in active-peak moving frame; duct profile is shown separately in panel E`.
- Longitudes exponenciales PDE: frente `4.993364` (`R2_log=0.989991`), cola `7.893627` (`R2_log=0.997987`), razon `0.632582`.
- Estilo panel B: `active pulse red solid; front-tail fit blue dashed; back-tail fit black dotted`.
- Estabilidad del colapso activo en panel B: `SEM_max=0.007312`, `SEM_mediana=0.000107`, calidad estimada `9.90/10`.
- Semillas BARW: `100`.
- Gramatica visual: color = observable fisico; estilo = procedencia (`PDE` solido, `BARW` discontinuo, teoria dash-dot).

## Salidas

- `figures/fig3_pde_barw_box_layout.png` y `.pdf`: composicion global.
- `figures/individual_panels/*.png` y `.pdf`: paneles individuales en box.
- `figures/quality_box/fig3_pde_barw_quality_box.png` y `.pdf`: caja compacta de metricas.
- `data/pde_two_species_*.csv`: datos PDE.
- `data/barw_ensemble_*.csv`: datos BARW.
- `data/barw_pde_profile_comparison.csv`: comparacion alineada por observable.
- `data/barw_profiles_moving_frame_collapsed.csv`: colapso BARW activo en marco movil.
- `data/pde_profiles_moving_frame_collapsed.csv`: colapso PDE activo en marco movil.
- `data/barw_duct_profiles_moving_frame_collapsed.csv`: colapso BARW ductal en marco movil.
- `data/pde_duct_profiles_moving_frame_collapsed.csv`: colapso PDE ductal en marco movil.
- `data/fig3_pde_barw_metrics.json`: metricas y configuracion reproducible.

## Nota sobre Fig. 3D experimental

Movie S1 es una pelicula suplementaria del modelo; se usa el frame con mayor senal roja y no sustituye los datos experimentales EdU de Fig. 3D.

Por tanto, esta salida debe citarse como analogia computacional Figure 3/S4. El panel Movie S1 queda como suplemento, no como reproduccion experimental completa de Fig. 3D.
