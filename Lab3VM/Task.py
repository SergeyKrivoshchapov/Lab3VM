import matplotlib.pyplot as plt
import numpy as np
import time

def lagrange(x_int, x_points, y_points):
    n = len(x_points) - 1
    result = 0.0
    for i in range(n + 1):
        fct = y_points[i]
        for j in range(n + 1):
            if j != i:
                fct *= (x_int - x_points[j]) / (x_points[i] - x_points[j])
        result += fct
    return result

def divided_differences(x_points, y_points):
    n = len(x_points)
    table = [list(y_points)]
    for k in range(1, n):
        prev = table[k - 1]
        row = [(prev[i + 1] - prev[i]) / (x_points[i + k] - x_points[i]) for i in range(n - k)]
        table.append(row)
    return [table[k][0] for k in range(n)]

def newton(x_int, x_points, coeff):
    n = len(coeff)
    result = coeff[0]
    fct = 1.0
    for k in range(1, n):
        fct *= (x_int - x_points[k - 1])
        result += coeff[k] * fct
    return result

def get_coefficients(x_points, y_points, verbose=False):
    A = np.vander(x_points, increasing=True)
    
    if verbose:
        print("\nМатрица Вандермонда:")
        for i in range(len(x_points)):
            print(f"x={x_points[i]}: ", end="")
            for j in range(len(x_points)):
                print(f"{A[i, j]:12.2f}", end="")
            print(f"  |  {y_points[i]:8.2f}")
        
        print("\nСЛАУ для поиска коэффициентов интерполяционного многочлена:")
        n = len(x_points)
        for i in range(n):
            equation = ""
            for j in range(n):
                coeff = A[i, j]
                if j == 0:
                    equation += f"{coeff:.0f}·a[{j}]"
                else:
                    if coeff >= 0:
                        equation += f" + {coeff:.0f}·a[{j}]"
                    else:
                        equation += f" - {-coeff:.0f}·a[{j}]"
            equation += f" = {y_points[i]}"
            print(equation)
    
    coeffs = np.linalg.solve(A, y_points)
    
    if verbose:
        print("\nКоэффициенты интерполяционного многочлена:")
        for i, coeff in enumerate(coeffs):
            print(f"  a[{i}] = {coeff}")
    
    return coeffs

def eval_coefficients(x_int, coeffs):
    result = 0.0
    power = 1.0
    for c in coeffs:
        result += c * power
        power *= x_int
    return result

print("Выберите задание:")
print("1 - Задание 1 (интерполяция по таблице из пособия)")
print("2 - Задание 2 (интерполяция по одной из трёх заданных функций)")
choice = input("Введите номер задания (1 или 2): ")

if choice == "1":
    x_points = [5, 7, 9, 11, 12]
    y_points = [3, -2, -2, 4, 15]
    
    newton_coeff = divided_differences(x_points, y_points)
    poly_coeff = get_coefficients(x_points, y_points, verbose=True)
    
    x_plot = np.linspace(4.5, 12.2, 200)
    y_lagrange = [lagrange(x, x_points, y_points) for x in x_plot]
    y_newton = [newton(x, x_points, newton_coeff) for x in x_plot]
    y_coeff = [eval_coefficients(x, poly_coeff) for x in x_plot]
    
    plt.figure(figsize=(14, 9))
    plt.plot(x_plot, y_lagrange, 'r-', linewidth=10, label='Многочлен Лагранжа')
    plt.plot(x_plot, y_newton, 'b--', linewidth=5, label='Многочлен Ньютона')
    plt.plot(x_plot, y_coeff, 'k-.', linewidth=5, label='Многочлен по найденным коэффициентам')
    plt.plot(x_points, y_points, 'o', color='green', markersize=14, label='Исходные точки')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('P(x)', fontsize=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    for i, (x, y) in enumerate(zip(x_points, y_points)):
        plt.annotate(f'({x}, {y})', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=14)

    plt.show()

elif choice == "2":
    functions = {
        "1": (lambda x: np.exp(x), 'exp(x)', (0, 5)),
        "2": (lambda x: np.sin(x), 'sin(x)', (0, 2*np.pi)),
        "3": (lambda x: np.log(x), 'ln(x)', (0.1, 10))
    }
    
    continue_flag = True
    while continue_flag:
        print("\nВыберите функцию для интерполяции:")
        print("1 - exp(x)")
        print("2 - sin(x)")
        print("3 - ln(x)")
        func_choice = input("Введите номер функции (1, 2 или 3): ")
        
        if func_choice not in functions:
            print("Неверный выбор")
            continue
        
        func, name, (x_min, x_max) = functions[func_choice]
        
        print(f"\nФункция: {name}")
        
        n_points = int(input(f"Введите количество точек: "))
        step = float(input(f"Введите шаг: "))
        
        x_points = np.arange(x_min, x_min + n_points * step, step)
        y_points = func(x_points)
        
        x_plot = np.linspace(min(x_points), max(x_points), 500)
        y_true = func(x_plot)
        
        plt.figure(figsize=(16, 10))
        
        start_time = time.time()
        y_lagrange = [lagrange(x, x_points, y_points) for x in x_plot]
        lagrange_time = time.time() - start_time
        
        start_time = time.time()
        newton_coeff = divided_differences(x_points, y_points)
        y_newton = [newton(x, x_points, newton_coeff) for x in x_plot]
        newton_time = time.time() - start_time
        
        start_time = time.time()
        poly_coeff = get_coefficients(x_points, y_points, verbose=False)
        y_coeff = [eval_coefficients(x, poly_coeff) for x in x_plot]
        coeff_time = time.time() - start_time
        
        plt.plot(x_plot, y_true, 'k-', linewidth=12, label='Исходная функция', alpha=0.8)
        plt.plot(x_plot, y_lagrange, 'r-', linewidth=5, label='Интерполяционный многочлен Лагранжа', alpha=0.9)
        plt.plot(x_plot, y_newton, 'b--', linewidth=5, label='Интерполяционный многочлен Ньютона', alpha=0.9)
        plt.plot(x_plot, y_coeff, 'g-.', linewidth=5, label='Интерполяционный многочлен по коэффициентам', alpha=0.9)
        plt.plot(x_points, y_points, 'o', color='yellow', markersize=10, label='Исходная точка')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.title(f'Интерполяция функции {name} (точек: {n_points}, шаг: {step})', fontsize=20)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        print(f"Время выполнения метода Лагранжа: {lagrange_time:.6f} сек")
        print(f"Время выполнения метода Ньютона: {newton_time:.6f} сек")
        print(f"Время выполнения метода по коэффициентам: {coeff_time:.6f} сек")
        
        if n_points > 10:
            max_diff_lag = np.max(np.abs(y_true - y_lagrange))
            max_diff_new = np.max(np.abs(y_true - y_newton))
            max_diff_coeff = np.max(np.abs(y_true - y_coeff))
            print(f"Максимальная погрешность метода Лагранжа: {max_diff_lag:.6f}")
            print(f"Максимальная погрешность метода Ньютона: {max_diff_new:.6f}")
            print(f"Максимальная погрешность метода по коэффициентам: {max_diff_coeff:.6f}")
        
        plt.show()
        
        next_choice = input("\nХотите построить график для другой функции? (y/n): ")
        if next_choice.lower() != 'y':
            continue_flag = False

else:
    print("Неверный выбор")
