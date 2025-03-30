import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Параметры сетки
nr, nphi = 50, 50  # Количество узлов по r и phi
r_min, r_max = 0, 1
phi_min, phi_max = 0, np.pi / 2
r = np.linspace(r_min, r_max, nr)
phi = np.linspace(phi_min, phi_max, nphi)
R, Phi = np.meshgrid(r, phi, indexing='ij')
dr = (r_max - r_min) / (nr - 1)
dphi = (phi_max - phi_min) / (nphi - 1)

# Инициализация матрицы и вектора правой части
N = nr * nphi
A = lil_matrix((N, N))
b = np.zeros(N)

def idx(i, j):
    return i * nphi + j

# Заполняем систему
for i in range(nr):
    for j in range(nphi):
        index = idx(i, j)
        r_i = r[i]
        
        if i == 0:  # Граничное условие u|_{t=0} = 0
            A[index, index] = 1
            b[index] = 0
        elif i == nr - 1:  # Граничное условие \frac{\partial u}{\partial r}|_{r=1} = sin(2φ)
            A[index, index] = 1 / dr
            A[index, idx(i - 1, j)] = -1 / dr
            b[index] = np.sin(2 * phi[j])
        elif j == 0:  # Граничное условие u|_{φ=0} = 0
            A[index, index] = 1
            b[index] = 0
        elif j == nphi - 1:  # Граничное условие \frac{\partial u}{\partial \varphi}|_{φ=π/2} = 0
            A[index, index] = 1
            A[index, idx(i, j - 1)] = -1
            b[index] = 0
        else:
            A[index, index] = -2 / dr**2 - 2 / (r_i**2 * dphi**2)
            A[index, idx(i - 1, j)] = 1 / dr**2 - 1 / (r_i * dr)
            A[index, idx(i + 1, j)] = 1 / dr**2 + 1 / (r_i * dr)
            A[index, idx(i, j - 1)] = 1 / (r_i**2 * dphi**2)
            A[index, idx(i, j + 1)] = 1 / (r_i**2 * dphi**2)
            b[index] = 0

# Решение системы
A = csr_matrix(A)
u = spsolve(A, b)
U = u.reshape(nr, nphi)

# Визуализация
X, Y = R * np.cos(Phi), R * np.sin(Phi)
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, U, levels=20, cmap='viridis')
plt.colorbar(contour, label='Значение u(r,φ)')
plt.title('Решение уравнения Лапласа в секторе круга')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('FOURTH_PROBLEM/OSADA/laplace_sector_solution.png', dpi=300, bbox_inches='tight')
plt.close()

print("Решение сохранено в файл 'FOURTH_PROBLEM/OSADA/laplace_sector_solution.png'")
