import numpy as np
from NewtonRaphson import *

def gauss_seidel_solver(A, b, x0=None, max_iter=10000, tol=1e-6):
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()

    A = A.tocsr()
    r = b - A @ x
    print("Norma inicial:", np.linalg.norm(r, np.inf))

    for k in range(max_iter):
        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i+1]
            Ai = A.indices[row_start:row_end]
            Av = A.data[row_start:row_end]

            suma = 0.0
            diag = 0.0
            for idx, j in enumerate(Ai):
                if j == i:
                    diag = Av[idx]
                else:
                    suma += Av[idx] * x[j]

            if diag != 0:
                x[i] = (b[i] - suma) / diag

        # Cálculo del residuo
        r = b - A @ x
        if np.linalg.norm(r, np.inf) < tol:
            print(f"Gauss-Seidel convergió en {k+1} iteraciones internas.")
            break

    print("Residuo final Gauss-Seidel:", np.linalg.norm(r, np.inf))
    return x


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
delta_V_test = gauss_seidel_solver(J_test, -F_val_test, x0=np.zeros_like(F_val_test), max_iter=10000, tol=1e-6)

# Aplicar la corrección como se haría en Newton-Raphson
V_corrected = V.flatten() + delta_V_test