import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import re


def parse_problem_file(filename):
    """Парсит файл с параметрами задачи"""
    params = {
        "a": 0.0,
        "b": 1.0,
        "alpha": 0.0,
        "beta": 1.0,
        "ya": 0.0,
        "yb": 0.0,
        "tolerance": 1e-4,
        "initial_step": 0.1,
        "p": "0",
        "q": "0",
        "r": "0",
    }

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()  # Удаляем комментарии
            if not line:
                continue

            # Ищем строки вида "param = value"
            match = re.match(r"([a-zA-Z_]+)\s*=\s*(.+)", line)
            if match:
                param, value = match.groups()
                param = param.strip()
                value = value.strip()
                if param in params:
                    try:
                        if param in ["p", "q", "r"]:
                            params[param] = value
                        else:
                            params[param] = float(value)
                    except ValueError:
                        print(
                            f"Warning: Не удалось распознать значение '{value}' для параметра {param}"
                        )

    return params


def solve_bvp(params):
    """Решает краевую задачу с заданными параметрами"""

    # Определяем функцию f(x, y, y_prime)
    def f(x, y, y_prime):
        return (
            -eval(params["p"], {"x": x}) * y_prime
            + eval(params["q"], {"x": x}) * y
            + eval(params["r"], {"x": x})
        )

    a = params["a"]
    b = params["b"]
    alpha = params["alpha"]
    beta = params["beta"]
    ya = params["ya"]
    yb = params["yb"]
    tol = params["tolerance"]
    h = params["initial_step"]

    # Адаптивное решение
    while True:
        x = np.linspace(a, b, max(3, int((b - a) / h) + 1))  # Минимум 3 точки
        n = len(x) - 1

        # Матрица коэффициентов
        main_diag = np.zeros(n + 1)
        lower_diag = np.zeros(n)
        upper_diag = np.zeros(n)
        rhs = np.zeros(n + 1)

        # Заполняем внутренние узлы
        for i in range(1, n):
            h_i = x[i + 1] - x[i]
            h_im1 = x[i] - x[i - 1]
            avg_h = (h_i + h_im1) / 2

            # Коэффициенты разностной схемы
            x_i = x[i]
            p_val = eval(params["p"], {"x": x_i})
            q_val = eval(params["q"], {"x": x_i})
            r_val = eval(params["r"], {"x": x_i})

            main_diag[i] = -2 / (h_i * h_im1) + q_val
            lower_diag[i - 1] = 1 / (h_im1 * avg_h) - p_val / (h_i + h_im1)
            upper_diag[i] = 1 / (h_i * avg_h) + p_val / (h_i + h_im1)
            rhs[i] = r_val

        # Левое граничное условие (alpha*y + beta*y' = ya)
        h0 = x[1] - x[0]
        main_diag[0] = alpha - beta / h0
        upper_diag[0] = beta / h0
        rhs[0] = ya

        # Правое граничное условие (y = yb)
        main_diag[n] = 1
        rhs[n] = yb

        # Строим и решаем систему
        A = diags([main_diag, lower_diag, upper_diag], [0, -1, 1], format="csc")
        y = spsolve(A, rhs)

        # Проверка точности (упрощенная)
        if h < tol / 10:
            break

        h /= 2

    return x, y


def save_results(x, y, filename):
    """Сохраняет результаты в файл"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Решение краевой задачи:\n")
        f.write("x\t\ty(x)\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.6f}\t{yi:.6f}\n")

    # График решения
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "b-", linewidth=2)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y(x)", fontsize=12)
    plt.title("Решение краевой задачи", fontsize=14)
    plt.grid(True)
    plt.savefig("SECOND_PROBLEM_boundary_value/solution_plot.png")
    plt.close()


def main():
    print("Решение краевой задачи методом конечных разностей")

    # Чтение параметров из файла
    try:
        params = parse_problem_file("SECOND_PROBLEM_boundary_value/problem.txt")
    except FileNotFoundError:
        print("Ошибка: файл problem.txt не найден!")
        return

    # Решение задачи
    x, y = solve_bvp(params)

    # Вывод результатов
    print("\nРезультаты:")
    print(f"Интервал: [{params['a']}, {params['b']}]")
    print(f"Количество узлов: {len(x)}")
    print(f"Шаг сетки: {x[1]-x[0]:.6f}")
    print("\nПервые 5 значений:")
    for i in range(min(5, len(x))):
        print(f"x = {x[i]:.6f}\ty = {y[i]:.6f}")

    # Сохранение результатов
    save_results(x, y, "SECOND_PROBLEM_boundary_value/solution.txt")
    print("\nРезультаты сохранены в файлы:")
    print("- solution.txt (таблица значений)")
    print("- solution_plot.png (график решения)")


if __name__ == "__main__":
    main()
