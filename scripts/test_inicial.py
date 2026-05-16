import numpy as np
from src.malla import crear_malla
from src.calor import dato_inicial_seno
from src.calor import solucion_exacta_seno
from src.calor import resolver_calor_euler_explicito

malla = crear_malla(0.0, 1.0, 6, 0.0, 0.1,11)

#print("x =", malla.x)
#print("dx =", malla.dx)
#print("t =", malla.t)
#print("dt =", malla.dt)

u0 = dato_inicial_seno(malla.x)

#print(u0)

#usamos una matriz U[n,j] para almacenar la solución en cada instante de tiempo y en todos los nodos espaciales
#n = tiempo, j = espacio
U = np.zeros((malla.nt, malla.nx))
U[0, :] = u0 #fila cero, todas las columnas -> solución en el instante inicial para todos los puntos espaciales
#print("shape de U:", U.shape)
#print("primera fila de U:", U[0, :])
#print("segunda fila de U:", U[1, :])

#ECUACIÓN DE CALOR CON EULER EXPLICITO
#U[n+1, j] = U[n, j] + lambda* (U[n, j-1] - 2*U[n, j] + U[n, j+1]) 
#lambda = D * dt / dx^2
D = 1.0
lambda_ = D * malla.dt / malla.dx**2 #debe ser menor o igual a 0.5 para estabilidad del método explícito
print("lambda =", lambda_)

#ajustamos los valores de la malla para que se cumpla la condición de estabilidad

#probamos a hacer un paso temporal
#U[1,0] = 0.0 #condición de frontera izquierda
#U[1,-1] = 0.0 #condición de frontera derecha
#for j in range(1, malla.nx-1): #evitar los extremos porque ya los hemos fijado con las condiciones de frontera
#    U[1, j] = U[0, j] + lambda_ * (U[0, j-1] - 2*U[0, j] + U[0, j+1])

#print("fila inicial U[0,:] =", U[0, :])
#print("fila tras un paso U[1,:] =", U[1, :])

#damos todos los pasos temporales
#for n in range(malla.nt-1):
#    U[n+1,0] = 0.0 #condición de frontera izquierda
#    U[n+1,-1] = 0.0 #condición de frontera derecha
#    for j in range(1, malla.nx-1): #evitar los extremos porque ya los hemos fijado con las condiciones de frontera
#        U[n+1, j] = U[n, j] + lambda_ * (U[n, j-1] - 2*U[n, j] + U[n, j+1])

#print("solución final U[-1,:] =", U[-1, :])

#comparamos con la solución exacta
#sol_exacta = solucion_exacta_seno(malla.x, malla.tf, D)
#print("solución exacta =", sol_exacta)
#print("error máximo =", np.max(np.abs(U[-1, :] - sol_exacta))) #para la convergencia

#probamos algunos casos más para comprobar el error máximo
casos = [
    (6, 11),
    (11, 41),
    (21, 161),
]

errores = []

for nx, nt in casos:
    malla = crear_malla(0.0, 1.0, nx, 0.0, 0.1, nt)
    u0 = dato_inicial_seno(malla.x)
    U = resolver_calor_euler_explicito(malla, D, u0)
    sol_exacta = solucion_exacta_seno(malla.x, malla.tf, D)
    error_maximo = np.max(np.abs(U[-1, :] - sol_exacta))
    errores.append(error_maximo)
    print(f"nx={nx}, nt={nt}, error máximo={error_maximo:.6e}")

#resultados observados:
#nx=6, nt=11, error máximo=5.861841e-03
#nx=11, nt=41, error máximo=1.519636e-03
#nx=21, nt=161, error máximo=3.786093e-04
#cuanto más refinada la malla (más nodos espaciales y temporales), menor el error máximo, 
# lo que indica que el método converge a la solución exacta.


for k in range(len(errores) - 1):
    p = np.log2(errores[k] / errores[k + 1])
    print(f"orden observado entre caso {k+1} y {k+2}: {p:.6f}") #calculamos el orden de convergencia observado entre cada par de casos consecutivos
