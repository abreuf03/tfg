import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.malla import crear_malla
from src.fisherkpp import (
    dato_inicial_fisher_escalon,
    resolver_fisher_kpp_euler_explicito,
    posicion_frente_por_nivel,
)

# Crear carpeta de resultados si no existe
Path("resultados").mkdir(exist_ok=True)

# Parámetros
D = 1.0
r = 1.0
K = 1.0
x0=2.0

# Malla
malla = crear_malla(0.0, 80.0, 801, 0.0, 20.0, 8001)

# Dato inicial y resolución numérica
u0 = dato_inicial_fisher_escalon(malla.x, K, x0)
U = resolver_fisher_kpp_euler_explicito(malla, D, r, K, u0)

print("mínimo final =", np.min(U[-1, :]))
print("máximo final =", np.max(U[-1, :]))

# -------------------------------------------------------------------
# Figura 1: perfiles de la solución
# -------------------------------------------------------------------

indice_inicial = 0
indice_intermedio = malla.nt // 2
indice_final = -1

t_inicial = malla.t[indice_inicial]
t_intermedio = malla.t[indice_intermedio]
t_final = malla.t[indice_final]

plt.figure(figsize=(8, 5))

plt.plot(
    malla.x,
    U[indice_inicial, :],
    label=fr"perfil inicial, $t={t_inicial:.1f}$"
)

plt.plot(
    malla.x,
    U[indice_intermedio, :],
    label=fr"perfil intermedio, $t={t_intermedio:.1f}$"
)

plt.plot(
    malla.x,
    U[indice_final, :],
    label=fr"perfil final, $t={t_final:.1f}$"
)

plt.xlabel(r"$x$ (adimensional)")
plt.ylabel(r"$u(x,t)$ (adimensional)")
plt.title("Perfiles de la solución de Fisher--KPP con Euler explícito")
plt.legend()
plt.grid(True)

plt.savefig(
    "resultados/fisher_kpp_euler_perfiles.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -------------------------------------------------------------------
# Cálculo de la posición del frente
# -------------------------------------------------------------------

nivel = K / 2
posiciones = []
tiempos = []

for n in range(malla.nt):
    xf = posicion_frente_por_nivel(malla.x, U[n, :], nivel)
    if xf is not None:
        posiciones.append(xf)
        tiempos.append(malla.t[n])

posiciones = np.array(posiciones)
tiempos = np.array(tiempos)

print("primeras posiciones del frente =", posiciones[:10])
print("velocidad teórica =", 2 * np.sqrt(r * D))

for t_corte in [4.0, 6.0, 8.0]:
    mascara = tiempos > t_corte
    coef = np.polyfit(tiempos[mascara], posiciones[mascara], 1)
    v_num = coef[0]
    print(f"Euler explícito, ajuste para t > {t_corte}: v = {v_num:.6f}")

# -------------------------------------------------------------------
# Figura 2: posición del frente
# -------------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    tiempos,
    posiciones,
    "o",
    label=fr"posición numérica del frente, $u=K/2={nivel:.1f}$"
)

mascara = tiempos > 6.0

if np.any(mascara):
    coef = np.polyfit(tiempos[mascara], posiciones[mascara], 1)
    v_num = coef[0]
    ordenada = coef[1]

    print("velocidad numérica del frente =", v_num)
    print("ordenada =", ordenada)

    plt.plot(
        tiempos[mascara],
        v_num * tiempos[mascara] + ordenada,
        label=fr"ajuste lineal para $t>6$, $v={v_num:.4f}$"
    )

plt.xlabel(r"$t$ (adimensional)")
plt.ylabel(r"$x_f(t)$ (adimensional)")
plt.title("Posición del frente de Fisher--KPP con Euler explícito")
plt.legend()
plt.grid(True)

plt.savefig(
    "resultados/fisher_kpp_euler_posicion_frente.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()