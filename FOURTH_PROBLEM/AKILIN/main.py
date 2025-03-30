import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Параметры области
x_min, x_max = -4, 4
y_min, y_max = 0, 3
nx, ny = 100, 75  # Количество точек по x и y
dx = (x_max - x_min) / (nx - 1)
dy = (y_max - y_min) / (ny - 1)

# Создаем сетку
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Определяем маску для эллипса
ellipse_mask = ((X-1)**2/4 + 2*(Y-1)**2 <= 1)

# Инициализация матрицы и правой части
N = nx * ny
A = lil_matrix((N, N))
b = np.zeros(N)

# Функция для индексации
def idx(i, j):
    return i * nx + j

# Заполняем матрицу и правую часть
for i in range(ny):
    for j in range(nx):
        # Пропускаем точки внутри эллипса
        if ellipse_mask[i, j]:
            A[idx(i,j), idx(i,j)] = 1
            b[idx(i,j)] = 0
            continue
            
        # Граничные условия
        if j == 0:  # x = -4 (левая граница)
            A[idx(i,j), idx(i,j)] = 1
            b[idx(i,j)] = y[i]  # Из условия задачи
        elif j == nx-1:  # x = 4 (правая граница)
            A[idx(i,j), idx(i,j-1)] = -1
            A[idx(i,j), idx(i,j)] = 1
            b[idx(i,j)] = 0  # Нейман (производная = 0)
        elif i == 0:  # y = 0 (нижняя граница)
            A[idx(i,j), idx(i,j)] = 1
            b[idx(i,j)] = 16 - x[j]**2
        elif i == ny-1:  # y = 3 (верхняя граница)
            A[idx(i,j), idx(i,j)] = 1
            b[idx(i,j)] = x[j] - 1
        else:  # Внутренние точки
            A[idx(i,j), idx(i,j)] = -2/dx**2 - 2/dy**2
            A[idx(i,j), idx(i,j-1)] = 1/dx**2
            A[idx(i,j), idx(i,j+1)] = 1/dx**2
            A[idx(i,j), idx(i-1,j)] = 1/dy**2
            A[idx(i,j), idx(i+1,j)] = 1/dy**2
            b[idx(i,j)] = x[j]**2 + 2*y[i]**2  # Правая часть уравнения

# Решаем систему
A = csr_matrix(A)
u = spsolve(A, b)

# Преобразуем решение в 2D массив
U = u.reshape(ny, nx)

# Маскируем область внутри эллипса
U[ellipse_mask] = np.nan

# Визуализация
plt.figure(figsize=(12, 8))
contour = plt.contourf(X, Y, U, levels=20, cmap=cm.viridis)
plt.colorbar(contour, label='Значение u(x,y)')
plt.contour(X, Y, ellipse_mask, levels=[0.5], colors='red')
plt.title('Решение уравнения Пуассона в прямоугольнике с эллиптическим вырезом')
plt.xlabel('x')
plt.ylabel('y')

# Сохраняем результат в файл
plt.savefig('FOURTH_PROBLEM\AKILIN\poisson_solution.png', dpi=300, bbox_inches='tight')
plt.close()

print("Решение сохранено в файл 'poisson_solution.png'")