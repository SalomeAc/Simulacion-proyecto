import numpy as np
from scipy.sparse import lil_matrix
from Jacobi import jacobi_solver
from GaussSeidel import gauss_seidel_solver
from Richardson import richardson_solver
from NewtonRaphson import compute_F, Jacobiano

# Parámetros
nx, ny = 200, 20
v_x, v_y = 1, 0.5
tol = 1e-6
max_iter = 100

# Estado inicial
V_init = np.full((nx, ny), 0.3)
V_init[0, :] = 1.0
V_init[0, 0] = V_init[0, ny - 1] = 0.0
V_init[:, 0] = V_init[:, ny - 1] = 0.0
V_init[-1, :] = 0.0
V_flat = V_init.flatten()

metodos_lineales = {
    "Jacobi": jacobi_solver,
    "Gauss-Seidel": gauss_seidel_solver,
    "Richardson": richardson_solver
}

for nombre, metodo in metodos_lineales.items():
    print(f"\n--- Método: {nombre} ---")
    V_flat_k = V_flat.copy()

    for k in range(max_iter):
        F_val = compute_F(V_flat_k, nx, ny, v_x, v_y)
        norma = np.linalg.norm(F_val, np.inf)
        print(f"Iteración {k}: ||F|| = {norma:.2e}")

        if norma < tol:
            print("Convergencia alcanzada!")
            break

        J = Jacobiano(V_flat_k, nx, ny, v_x, v_y)
        x0 = np.zeros_like(F_val)
        delta_V = metodo(J, -F_val, x0, max_iter=1000, tol=tol)  # Método iterativo
        V_flat_k += delta_V

    else:
        print("No convergió dentro del máximo de iteraciones")
