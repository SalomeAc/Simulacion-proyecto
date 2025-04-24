import numpy as np
import scipy.sparse as sp
from NewtonRaphson import *

def jacobi_solver(A, b, x0=None, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()

    # Verificar que la diagonal no tenga ceros para evitar divisiones por cero
    D = A.diagonal()
    if np.any(D == 0):
        raise ValueError("La matriz tiene ceros en la diagonal, Jacobi no es aplicable directamente.")

    D_inv = 1.0 / D

    print("Norma del residuo inicial:", np.linalg.norm(b - A @ x, np.inf))

    for k in range(max_iter):
        r = b - A @ x
        delta_x = D_inv * r
        x += delta_x

        if np.linalg.norm(delta_x, np.inf) < tol:
            print(f"Jacobi convergió en {k} iteraciones internas.")
            break

    r_norm = np.linalg.norm(b - A @ x, np.inf)
    print("Residuo final Jacobi:", r_norm)

    return x

# Prueba del método de Jacobi sobre un sistema nuevo (como parte del proceso de Newton-Raphson)
V = np.full((nx, ny), 0.3)  # Valor inicial de V

# Condiciones de frontera
V[0, :] = 1.0          # Frontera izquierda (columna 0) a 1
V[0, 0] = 0.0          # Esquina inferior izquierda a 0
V[0, ny-1] = 0.0       # Esquina superior izquierda a 0
V[:, 0] = 0.0          # Frontera inferior (fila 0) a 0
V[:, ny-1] = 0.0       # Frontera superior (última fila) a 0
V[-1, :] = 0.0         # Frontera derecha a 0
F_val_test = compute_F(V.flatten(), nx, ny, v_x, v_y)
J_test = Jacobiano(V.flatten(), nx, ny, v_x, v_y)

print("Norma inicial:", np.linalg.norm(J_test @ np.zeros_like(F_val_test) + F_val_test, np.inf))
delta_V_test = jacobi_solver(J_test, -F_val_test, x0=np.zeros_like(F_val_test), max_iter=1000, tol=1e-6)

# Aplicar la corrección como se haría en Newton-Raphson
V_corrected = V.flatten() + delta_V_test

