import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd

# Parámetros de la malla
nx, ny = 200, 20  # Tamaño de la malla
v_x = 1           # Parámetro de velocidad
v_y = 0.5         # Parámetro de vorticidad
tolerance = 1e-6  # Tolerancia para la convergencia
max_iter = 100     # Máximo número de iteraciones de Newton

# Inicializar la matriz de potencial
V = np.full((nx, ny), 0.3)  # Valor inicial de V

# Condiciones de frontera
V[0, :] = 1.0          # Frontera izquierda (columna 0) a 1
V[0, 0] = 0.0          # Esquina inferior izquierda a 0
V[0, ny-1] = 0.0       # Esquina superior izquierda a 0
V[:, 0] = 0.0          # Frontera inferior (fila 0) a 0
V[:, ny-1] = 0.0       # Frontera superior (última fila) a 0
V[-1, :] = 0.0         # Frontera derecha a 0

# Función que calcula F(V) para el método de Newton-Raphson
# La función F(V) representa el sistema de ecuaciones no lineales
def compute_F(V_flat, nx, ny, v_x, v_y):
    V = V_flat.reshape((nx, ny))
    F_val = np.zeros_like(V)

    for i in range(1, nx-1):
        for j in range(1, ny-1):

            # Aquí se hace no lineal
            termino1 = V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1]
            termino2 = (1/2)*v_x*V[i,j]*(V[i+1,j] - V[i-1,j])
            termino3 = (1/2)*v_y*V[i,j]*(V[i,j+1] - V[i,j-1])
            F_val[i,j] = V[i,j] - 0.25 * (termino1 - termino2 - termino3)
    
    # Aplicar condiciones de frontera 
    F_val[0, :] = V[0, :] - 1.0
    F_val[0, 0] = V[0, 0]
    F_val[0, ny-1] = V[0, ny-1]
    F_val[:, 0] = V[:, 0]
    F_val[:, ny-1] = V[:, ny-1]
    F_val[-1, :] = V[-1, :]
    
    return F_val.flatten()

# Función para calcular el Jacobiano analíticamente
def Jacobiano(V_flat, nx, ny, v_x, v_y, h=1.0):
    N = nx * ny
    J = lil_matrix((N, N))
    V = V_flat.reshape((nx, ny))

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            idx = i*ny + j

            # Término principal: ∂F/∂V[i,j]
            d_adv_x = (1/2)*v_x*(V[i+1,j] - V[i-1,j])
            d_adv_y = (1/2)*v_y*(V[i,j+1] - V[i,j-1])
            J[idx, idx] = 1 + 0.25 * (d_adv_x + d_adv_y)

            # Derivadas respecto a vecinos
            J[idx, (i+1)*ny + j] = -0.25 * (1 - (1/2)*v_x*V[i,j])
            J[idx, (i-1)*ny + j] = -0.25 * (1 + (1/2)*v_x*V[i,j])
            J[idx, i*ny + (j+1)] = -0.25 * (1 - (1/2)*v_y*V[i,j])
            J[idx, i*ny + (j-1)] = -0.25 * (1 + (1/2)*v_y*V[i,j])
    
    # Fronteras con valores fijos (Dirichlet): ∂F/∂V = 1 para el mismo punto, 0 para otros
    # Frontera izquierda
    for j in range(ny):
        idx = j
        J[idx, :] = 0
        J[idx, idx] = 1
        
    
    # Frontera derecha
    for j in range(ny):
        idx = (nx-1)*ny + j
        J[idx, :] = 0
        J[idx, idx] = 1
    
    # Fronteras inferior y superior
    for i in range(1, nx-1):
        # Frontera inferior (j=0)
        idx = i*ny
        J[idx, :] = 0
        J[idx, idx] = 1
        
        # Frontera superior (j=ny-1)
        idx = i*ny + (ny-1)
        J[idx, :] = 0
        J[idx, idx] = 1
    
    return J.tocsr()

# Aplanar el array inicial
V_flat = V.flatten()

# Método de Newton-Raphson
for k in range(max_iter):
    F_val = compute_F(V_flat, nx, ny, v_x, v_y)
    residual_norm = np.linalg.norm(F_val, np.inf) # norma infinito
    print(f"Iteración {k}: Residuo = {residual_norm:.6f}")
    
    if residual_norm < tolerance:
        print("Convergencia alcanzada!")
        break
    
    # Calcular Jacobiano analíticamente
    J = Jacobiano(V_flat, nx, ny, v_x, v_y)
    
    # Resolver sistema lineal J ΔV = -F(V)
    delta_V = spsolve(J, -F_val)
    
    # Actualizar solución
    V_flat += delta_V

final_residual = np.linalg.norm(compute_F(V_flat, nx, ny, v_x, v_y), np.inf)
print(f"Residuo final verificado: {final_residual}")

# Reconstruir la matriz 2D
V = V_flat.reshape((nx, ny))

# Guardar matrices en un archivo Excel
def save_to_excel(V):
    # Transponer la matriz para que coincida con la orientación del gráfico
    V_transposed = V.T
    
    # Crear un DataFrame de pandas
    df_V = pd.DataFrame(V_transposed)
    
    # También guardamos una matriz de ceros para vy como en el código original
    vy_vals = np.full((ny, nx), v_y)
    df_vy = pd.DataFrame(vy_vals)
    
    # Guardar en Excel
    with pd.ExcelWriter("flujo_resultados.xlsx") as writer:
        df_V.to_excel(writer, sheet_name="Velocidad_x", index=False)
        df_vy.to_excel(writer, sheet_name="Velocidad_y", index=False)
    
    print("Matrices guardadas en 'flujo_resultados.xlsx'")

# Guardar en Excel
save_to_excel(V)

# Mapa de calor
plt.figure(figsize=(15, 5))
plt.imshow(V.T, cmap=cm.hot, origin='lower', aspect='auto', interpolation='bilinear')
plt.colorbar(label='Potencial')
plt.title(f'Mapa de Calor de velocidad (v_x = {v_x}, v_y = {v_y})')
plt.xlabel('Dirección X')
plt.ylabel('Dirección Y')
plt.savefig('mapa_calor_potencial.png', dpi=300, bbox_inches='tight')
print("Mapa de calor guardado como 'mapa_calor_potencial.png'")
plt.show()

# VERIFICACIÓN DE SIMETRIÍA DEL JACOBIANO

# Extrae un bloque 200×200 del Jacobiano
block_size = 200
J_block = J[:block_size, :block_size].todense()

# Muestra en pantalla
print(np.round(J_block, 3))

# verlo en Excel:
pd.DataFrame(np.round(J_block, 6)).to_excel(
    "jacobiano_bloque_200x200.xlsx", index=False, header=False)
print("Bloque 200×200 guardado en jacobiano_bloque_200x200.xlsx")