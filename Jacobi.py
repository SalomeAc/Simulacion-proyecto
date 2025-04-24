from NewtonRaphson import *

max_iter = 10000

# Método de Jacobi
converged = False
V_new = np.copy(V)

for k in range(max_iter):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            laplaciano = V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1]
            adv_x = 0.5 * v_x * (V[i+1, j] - V[i-1, j])
            adv_y = 0.5 * v_y * (V[i, j+1] - V[i, j-1])
            V_new[i, j] = 0.25 * (laplaciano - adv_x - adv_y)

    # Condiciones de frontera (reaplicar en cada paso)
    V_new[0, :] = 1.0
    V_new[0, 0] = 0.0
    V_new[0, ny-1] = 0.0
    V_new[:, 0] = 0.0
    V_new[:, ny-1] = 0.0
    V_new[-1, :] = 0.0

    # Verificación de convergencia
    delta = np.max(np.abs(V_new - V))
    print(f"Iteración {k}: Cambio máximo = {delta:.6e}")
    if delta < tolerance:
        print(f"¡Jacobi convergió en {k} iteraciones!")
        converged = True
        break

    V[:] = V_new[:]

if not converged:
    print("Jacobi no alcanzó la convergencia")

# Mostrar el resultado
plt.figure(figsize=(15, 5))
plt.imshow(V.T, cmap=cm.hot, origin='lower', aspect='auto')
plt.colorbar(label='Potencial')
plt.title(f'Mapa de Calor del Potencial (Jacobi, v_x={v_x}, v_y={v_y})')
plt.xlabel('Dirección X')
plt.ylabel('Dirección Y')
plt.savefig("jacobi_mapa_calor.png", dpi=300, bbox_inches='tight')
plt.show()