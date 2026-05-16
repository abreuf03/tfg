import numpy as np
from src.malla import crear_malla
from src.calor import dato_inicial_seno
from src.calor import solucion_exacta_seno
from src.calor import resolver_calor_crank_nicolson



#malla = crear_malla(0.0, 1.0, 6, 0.0, 0.1, 11) #con estos datos sale peor que euler explícito
#D = 1.0
#u0 = dato_inicial_seno(malla.x)

#U_cn = resolver_calor_crank_nicolson(malla, D, u0)
#sol_exacta = solucion_exacta_seno(malla.x, malla.tf, D)

#print("solución CN final =", U_cn[-1, :])
#print("solución exacta   =", sol_exacta)
#print("error máximo CN   =", np.max(np.abs(U_cn[-1, :] - sol_exacta)))

casos = [
    (6, 11),
    (11, 41),
    (21, 161),
]

errores = []
D=1.0

for nx, nt in casos:
    malla = crear_malla(0.0, 1.0, nx, 0.0, 0.1, nt)
    u0 = dato_inicial_seno(malla.x)
    U = resolver_calor_crank_nicolson(malla, D, u0)
    sol_exacta = solucion_exacta_seno(malla.x, malla.tf, D)
    error_maximo = np.max(np.abs(U[-1, :] - sol_exacta))
    errores.append(error_maximo)
    print(f"nx={nx}, nt={nt}, error máximo={error_maximo:.6e}")

#resultados observados:
#nx=6, nt=11, error máximo=1.127712e-02
#nx=11, nt=41, error máximo=3.009367e-03
#nx=21, nt=161, error máximo=7.553402e-04

for k in range(len(errores) - 1):
    p = np.log2(errores[k] / errores[k + 1])
    print(f"orden observado entre caso {k+1} y {k+2}: {p:.6f}")