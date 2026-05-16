import numpy as np
import matplotlib.pyplot as plt

from src.malla import crear_malla
from src.fisherkpp import (
    dato_inicial_fisher_escalon,
    resolver_fisher_kpp_euler_explicito,
    posicion_frente_por_nivel
)

def velocidad_bramson(t0, T):
    return 2.0 - (3.0 * np.log(T / t0)) / (2.0 * (T - t0))


def ajustar_velocidad(t, xf, t0):
    mascara = t >= t0
    coef = np.polyfit(t[mascara], xf[mascara], 1)
    return coef[0], coef[1]

#parametros del problema
D = 1.0
r = 1.0
K = 1.0

x0 = 20.0 #para que no este tan cerca del origen y se pueda apreciar bien la evolución del frente
a = 0.0
b = 260.0

Tf = 100.0
nx = 1301
nt = 10001

malla = crear_malla(a, b, nx, 0.0, Tf, nt)
u0 = dato_inicial_fisher_escalon(malla.x, K, x0)
U = resolver_fisher_kpp_euler_explicito(malla, D, r, K, u0)
n_frente = 0.5 * K
xf = np.array([posicion_frente_por_nivel(malla.x, U[n, :], n_frente) for n in range(malla.nt)])

t_cortes = [20.0, 40.0, 50.0]
print("t0, v_num, v_bramson, error_rel_bramson")

for t0 in t_cortes:
    v_num, v_ajustada = ajustar_velocidad(malla.t, xf, t0)
    v_bramson = velocidad_bramson(t0, Tf)
    error_rel_bramson = abs(v_num - v_bramson) / abs(v_bramson)
    print(f"{t0:.1f}, {v_num:.4f}, {v_bramson:.4f}, {error_rel_bramson:.4e}")

plt.figure(figsize=(8, 5))
plt.plot(malla.t, xf, label="posición numérica del frente")

t0_fit = 40.0
v_num, b_fit = ajustar_velocidad(malla.t, xf, t0_fit)
plt.plot(
    malla.t,
    v_num * malla.t + b_fit,
    label=f"ajuste lineal desde t0={t0_fit:g}",
)

plt.xlabel("t (adimensional)")
plt.ylabel("x_f(t) (adimensional)")
plt.legend()
plt.tight_layout()
plt.savefig("resultados/fisher_kpp_bramson_T100.png", dpi=300)