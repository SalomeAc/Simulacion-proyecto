from NewtonRaphson import *

import numpy as np
import scipy.sparse as sp

def richardson_solver(A, b, x0=None, max_iter=10000, tol=1e-6):
    """
    Iteración fija x_{k+1} = (I - A)x_k + b  (Richardson con ω = 1).

    Devuelve
    --------
    x : ndarray        --  Solución aproximada.
    k : int            --  Número de iteraciones realizadas.
    """
    if sp.issparse(A):
        A = A.tocsr()

    n = b.size
    x = np.zeros(n) if x0 is None else x0.copy()
    I_minus_A = sp.eye(n, format='csr') - A

    for k in range(1, max_iter + 1):
        x = I_minus_A @ x + b
        r = b - A @ x
        if np.linalg.norm(r, np.inf) < tol:
            print(f"Richardson convergió en {k} iteraciones internas.")
            break
    else:
        print("Richardson alcanzó max_iter sin converger.")

    return x


V_test = np.full((nx, ny), 0.3)
F_val_test = compute_F(V_test.flatten(), nx, ny, v_x, v_y)
J_test = Jacobiano(V_test.flatten(), nx, ny, v_x, v_y)

print("Norma inicial:", np.linalg.norm(J_test @ np.zeros_like(F_val_test) + F_val_test, np.inf))
delta_V_test = richardson_solver(J_test, -F_val_test, x0=np.zeros_like(F_val_test), max_iter=10000, tol=1e-6)

# Aplicar la corrección como se haría en Newton-Raphson
V_corrected = V_test.flatten() + delta_V_test

