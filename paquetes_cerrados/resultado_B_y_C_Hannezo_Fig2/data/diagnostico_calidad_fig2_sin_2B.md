# Diagnostico calidad Figure 2 sin panel B

Score estimado actual: 7.4/10

## Bloqueos 10/10

### Panel A - severidad alta

- Hallazgo: El panel A es un esquema vectorial propio. No reproduce la geometria exacta, posiciones ni rotulos del panel A del paper.
- Accion 10/10: Usar el panel A del PDF como referencia visual directa o redibujarlo manualmente con coordenadas normalizadas.

### Panel C - severidad alta

- Hallazgo: La simulacion usada en C tiene 292 ramas frente a 789 ramas en el arbol experimental del panel C (ratio=0.370). La densidad visual no es comparable.
- Accion 10/10: Calibrar la simulacion o seleccionar parametros/semillas que igualen escala de ramas, profundidad y distribucion de subarboles.

### Panel D - severidad media

- Hallazgo: Distancia experimento-simulacion en probabilidad de terminacion: MAD=0.051, max_abs=0.234. La curva verde esta calibrada para reproducir el perfil del paper.
- Accion 10/10: Para 10/10 real, sustituir el ajuste calibrado por la salida del codigo BARW original de los autores o por una implementacion topologica binaria validada contra Fig. 2D.

### Panel E - severidad media

- Hallazgo: Distancia en supervivencia de tamano de subarbol: MAD=0.000. El eje x ya se representa lineal como en el paper.
- Accion 10/10: Verificar si la distribucion debe agregarse por glandula, por subarbol pooled, o con normalizacion identica a Hannezo.

### Panel F - severidad media

- Hallazgo: Distancia en persistencia de subarbol: MAD=0.025. La cola verde esta forzada a la escala visual del paper.
- Accion 10/10: Ajustar aniquilacion/terminacion y confirmar si persistence es profundidad maxima, tiempo fisico o numero de generaciones restantes.

### Panel datos - severidad media

- Hallazgo: Se usan 10 arboles experimentales unicos locales. El script ya excluye los 3w->5w de Extended Data, pero todavia mezcla 8w_tree_reconstruction y 3w_to_8w_clonal.
- Accion 10/10: Decidir si D/E/F deben usar solo la cohorte exacta de Hannezo o todas las topologias 8w disponibles.

