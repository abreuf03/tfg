# Benchmark de escalabilidad BARW

Este paquete mide la regla computacional mas costosa del modelo BARW:
detectar si una punta activa se encuentra dentro del radio de aniquilacion
de un conducto ya depositado.

## Ejecucion

```powershell
python benchmark_escalabilidad_barw.py
```

Modos:

```powershell
python benchmark_escalabilidad_barw.py --mode quick
python benchmark_escalabilidad_barw.py --mode full
```

## Que produce

- `data/benchmark_scaling.csv`: tiempos, aceleraciones y discrepancias.
- `data/benchmark_metrics.json`: metricas resumidas y checks de calidad.
- `data/summary_benchmark_escalabilidad_barw.md`: resumen para memoria.
- `figures/benchmark_escalabilidad_barw_layout.png/pdf`: figura compuesta.
- `figures/individual_panels/*.png/pdf`: paneles individuales.

## Uso en el TFG

Este script cubre la aportacion informatica: demostrar que la busqueda
espacial optimizada acelera el simulador BARW sin cambiar la regla fisica
de terminacion por proximidad.
