import numpy as np

from src.malla import crear_malla
from src.campo_medio.config import CampoMedioConfig
from src.campo_medio.solvers import (
    resolver_euler_explicito,
    resolver_imex_cn,
)


malla = crear_malla(
    a=0.0,
    b=20.0,
    nx=101,
    t0=0.0,
    tf=5.0,
    nt=5001,
)

config = CampoMedioConfig(
    D=1.0,
    rb=0.1,
    re=3.0,
    n0=1.0,
    a_in=1.0,
    a_out=0.0,
    a0=0.0,
    i0=0.0,
)

A_euler, I_euler = resolver_euler_explicito(
    malla,
    config,
)

A_cn, I_cn = resolver_imex_cn(
    malla,
    config,
)

print("Dimensiones Euler:")
print("A:", A_euler.shape)
print("I:", I_euler.shape)

print("\nDimensiones CN:")
print("A:", A_cn.shape)
print("I:", I_cn.shape)

print("\nExtremos de Euler:")
print("Frontera izquierda:", A_euler[:, 0])
print("Frontera derecha:", A_euler[:, -1])

print("\nValores mínimos y máximos:")
print("A Euler:", np.min(A_euler), np.max(A_euler))
print("I Euler:", np.min(I_euler), np.max(I_euler))
print("A CN:", np.min(A_cn), np.max(A_cn))
print("I CN:", np.min(I_cn), np.max(I_cn))

print("\n¿Hay NaN?")
print("Euler A:", np.isnan(A_euler).any())
print("Euler I:", np.isnan(I_euler).any())
print("CN A:", np.isnan(A_cn).any())
print("CN I:", np.isnan(I_cn).any())

error_A = np.max(np.abs(A_euler[-1] - A_cn[-1]))
error_I = np.max(np.abs(I_euler[-1] - I_cn[-1]))

print("\nDiferencia entre métodos en el tiempo final:")
print("A:", error_A)
print("I:", error_I)