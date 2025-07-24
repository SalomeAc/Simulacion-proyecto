import numpy as np
from scipy.sparse import lil_matrix
from Jacobi import jacobi_solver
from GaussSeidel import gauss_seidel_solver
from Richardson import richardson_solver
from NewtonRaphson import compute_F, Jacobiano, V
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib import cm

# Parámetros
nx, ny = 200, 20
v_x, v_y = 1, 0.5
tol = 1e-6
max_iter = 100
V_final = None 
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
        delta_V = metodo(J, -F_val, x0, max_iter=1000, tol=tol)  
        V_flat_k += delta_V

        if nombre == "Gauss-Seidel":
            V_final = V_flat_k.reshape((nx, ny))

    else:
        print("No convergió dentro del máximo de iteraciones")

# Mapa de calor final
nx, ny = V.shape
x = np.arange(nx)
y = np.arange(ny)

# Crear el spline con base en los datos originales
spline_interp = RectBivariateSpline(y, x, V_final.T, kx=3, ky=3)

# Redimensionar a una malla más fina para suavizar visualmente
x_fino = np.linspace(0, nx - 1, 400)
y_fino = np.linspace(0, ny - 1, 200)

# Evaluar el spline en la malla fina
V_fino = spline_interp(y_fino, x_fino).T

# Graficar el mapa de calor suavizado
plt.figure(figsize=(15, 5))
plt.imshow(V_fino.T, cmap=cm.hot, origin='lower', aspect='auto')
plt.colorbar(label='Velocidad')
plt.title(f'Mapa de Calor con Splines (v_x = {v_x}, v_y = {v_y})')
plt.xlabel('Dirección X')
plt.ylabel('Dirección Y')
plt.savefig('mapa_calor_splines.png', dpi=300, bbox_inches='tight')
plt.show()