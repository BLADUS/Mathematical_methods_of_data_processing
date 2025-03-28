import numpy as np
from pathlib import Path

def solve_tridiagonal_system(matrix, log_file):
    """
    Решает систему с трехдиагональной матрицей методом прогонки (Томаса)
    с логированием всех шагов
    
    Args:
        matrix (np.ndarray): Расширенная матрица системы (N x N+1)
        log_file (file): Файловый объект для записи логов
    
    Returns:
        np.ndarray: Вектор решений
    """
    n = matrix.shape[0]
    a = np.zeros(n-1)  # Альфа коэффициенты
    B = np.zeros(n)    # Бета коэффициенты
    x = np.zeros(n)    # Решения
    
    log_file.write("=== ПРЯМОЙ ХОД ===\n")
    
    # Инициализация первого шага
    y = matrix[0, 0]
    a[0] = -matrix[0, 1] / y
    B[0] = matrix[0, -1] / y
    
    log_file.write(f"\nШаг 0 (инициализация):\n")
    log_file.write(f"y[0] = A[0,0] = {matrix[0,0]}\n")
    log_file.write(f"a[0] = -A[0,1]/y[0] = -{matrix[0,1]}/{y} = {a[0]:.6f}\n")
    log_file.write(f"B[0] = A[0,{n}]/y[0] = {matrix[0,-1]}/{y} = {B[0]:.6f}\n")
    
    # Прямой ход
    for i in range(1, n):
        log_file.write(f"\nШаг {i}:\n")
        
        y_prev = y
        y = matrix[i, i] + matrix[i, i-1] * a[i-1]
        
        log_file.write(f"y[{i}] = A[{i},{i}] + A[{i},{i-1}]*a[{i-1}] = ")
        log_file.write(f"{matrix[i,i]} + {matrix[i,i-1]}*{a[i-1]:.6f} = {y:.6f}\n")
        
        if i < n-1:
            a_prev = a[i] if i < len(a) else 0
            a[i] = -matrix[i, i+1] / y
            log_file.write(f"a[{i}] = -A[{i},{i+1}]/y[{i}] = -{matrix[i,i+1]}/{y:.6f} = {a[i]:.6f}\n")
        
        B_prev = B[i-1]
        B[i] = (matrix[i, -1] - matrix[i, i-1] * B[i-1]) / y
        log_file.write(f"B[{i}] = (A[{i},{n}] - A[{i},{i-1}]*B[{i-1}])/y[{i}] = ")
        log_file.write(f"({matrix[i,-1]} - {matrix[i,i-1]}*{B_prev:.6f})/{y:.6f} = {B[i]:.6f}\n")
    
    log_file.write("\n=== ОБРАТНЫЙ ХОД ===\n")
    
    # Обратный ход
    x[-1] = B[-1]
    log_file.write(f"\nx[{n-1}] = B[{n-1}] = {B[-1]:.6f}\n")
    
    for i in range(n-2, -1, -1):
        x_prev = x[i+1]
        x[i] = a[i] * x[i+1] + B[i]
        log_file.write(f"x[{i}] = a[{i}]*x[{i+1}] + B[{i}] = ")
        log_file.write(f"{a[i]:.6f}*{x_prev:.6f} + {B[i]:.6f} = {x[i]:.6f}\n")
    
    return x

def read_matrix_from_file(file_path):
    """Читает матрицу из файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        n = int(f.readline())
        matrix = []
        for _ in range(n):
            row = list(map(float, f.readline().split()))
            matrix.append(row)
        return np.array(matrix)

def input_matrix_manually():
    """Позволяет ввести матрицу вручную"""
    n = int(input("Введите количество уравнений (и переменных): "))
    print(f"Введите расширенную матрицу {n}x{n+1} (по строкам, элементы через пробел):")
    matrix = []
    for i in range(n):
        while True:
            row = input(f"Строка {i+1}: ").split()
            if len(row) == n + 1:
                try:
                    matrix.append(list(map(float, row)))
                    break
                except ValueError:
                    print("Ошибка: все элементы должны быть числами")
            else:
                print(f"Ошибка: нужно ввести {n+1} элементов")
    return np.array(matrix)

def main():
    print("Решение трехдиагональных систем уравнений методом прогонки")
    print("Выберите способ ввода матрицы:")
    print("1 - Из файла (matrix.txt)")
    print("2 - Вручную")
    choice = input("Ваш выбор (1/2): ")
    
    try:
        # Создаем папку для результатов, если ее нет
        Path("thomasAlgorithm").mkdir(exist_ok=True)
        
        if choice == '1':
            file_path = 'thomasAlgorithm/matrix.txt'
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Файл {file_path} не найден")
            matrix = read_matrix_from_file(file_path)
        elif choice == '2':
            matrix = input_matrix_manually()
        else:
            raise ValueError("Неверный выбор")
        
        n = matrix.shape[0]
        print(f"\nРазмер системы: {n} уравнений, {n} переменных")
        print("Введенная матрица:")
        print(matrix)
        
        # Открываем файл для логов с указанием кодировки UTF-8
        with open('thomasAlgorithm/answer.txt', 'w', encoding='utf-8') as log_file:
            log_file.write("=== НАЧАЛО РАСЧЕТА ===\n")
            log_file.write(f"Размер системы: {n} уравнений, {n} переменных\n")
            log_file.write("Исходная матрица:\n")
            np.savetxt(log_file, matrix, fmt='%.6f')
            log_file.write("\n")
            
            solution = solve_tridiagonal_system(matrix, log_file)
            
            log_file.write("\n=== РЕЗУЛЬТАТ ===\n")
            for i, x_val in enumerate(solution, 1):
                log_file.write(f"x{i} = {x_val:.6f}\n")
        
        print("\nРешение системы:")
        for i, x in enumerate(solution, 1):
            print(f"x{i} = {x:.6f}")
        
        print("\nПодробный отчет о решении сохранен в файл thomasAlgorithm/answer.txt")
        
    except Exception as e:
        print(f"\nОшибка: {e}")
        print("Пожалуйста, проверьте введенные данные и попробуйте снова")

if __name__ == "__main__":
    main()