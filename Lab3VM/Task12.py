import matplotlib.pyplot as plt
import numpy as np
import time

def lagrange(x_int, x_points, y_points):
    n = len(x_points) - 1
    result = 0.0
    for i in range(n + 1):
        term = y_points[i]
        for j in range(n + 1):
            if j != i:
                term *= (x_int - x_points[j]) / (x_points[i] - x_points[j])
        result += term
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
    term = 1.0
    for k in range(1, n):
        term *= (x_int - x_points[k - 1])
        result += coeff[k] * term
    return result

def get_coefficients(x_points, y_points):
    A = np.vander(x_points, increasing=True)
    return np.linalg.solve(A, y_points)

def eval_coefficients(x_int, coeffs):
    result = 0.0
    power = 1.0
    for c in coeffs:
        result += c * power
        power *= x_int
    return result

print("Выберите задание:")
print("1 - Задание 1 (интерполяция по таблице из пособия)")
print("2 - Задание 2 (интерполяция экспоненциальной функции)")
choice = input("Введите номер задания (1 или 2): ")

if choice == "1":
    x_points = [5, 7, 9, 11, 12]
    y_points = [3, -2, -2, 4, 15]
    
    newton_coeff = divided_differences(x_points, y_points)
    poly_coeff = get_coefficients(x_points, y_points)
    
    test_points = [8, 11, 13]
    for x in test_points:
        y = lagrange(x, x_points, y_points)
        print(f"L({x}) = {y:.6f}")
    
    for x in test_points:
        y = newton(x, x_points, newton_coeff)
        print(f"N({x}) = {y:.6f}")
    
    for x in test_points:
        y = eval_coefficients(x, poly_coeff)
        print(f"C({x}) = {y:.6f}")
    
    x_plot = np.linspace(4, 13, 200)
    y_lagrange = [lagrange(x, x_points, y_points) for x in x_plot]
    y_newton = [newton(x, x_points, newton_coeff) for x in x_plot]
    y_coeff = [eval_coefficients(x, poly_coeff) for x in x_plot]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_lagrange, 'b-', linewidth=2, label='Многочлен Лагранжа')
    plt.plot(x_plot, y_newton, 'g--', linewidth=4, label='Многочлен Ньютона')
    plt.plot(x_plot, y_coeff, 'm:', linewidth=5, label='Через коэффициенты')
    plt.plot(x_points, y_points, 'ro', markersize=8, label='Узлы интерполяции')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.legend()
    
    for i, (x, y) in enumerate(zip(x_points, y_points)):
        plt.annotate(f'({x}, {y})', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.show()

elif choice == "2":
    func = lambda x: np.exp(-x) * np.sin(2 * np.pi * x)
    name = 'exp(-x)*sin(2πx)'
    
    print(f"\n{name}")
    
    n_points = int(input(f"Введите количество точек: "))
    step = float(input(f"Введите шаг: "))
    
    x_points = np.arange(0, n_points * step, step)
    y_points = func(x_points)
    
    x_plot = np.linspace(min(x_points), max(x_points), 500)
    y_true = func(x_plot)
    
    plt.figure(figsize=(14, 8))
    
    start_time = time.time()
    y_lagrange = [lagrange(x, x_points, y_points) for x in x_plot]
    lagrange_time = time.time() - start_time
    
    start_time = time.time()
    newton_coeff = divided_differences(x_points, y_points)
    y_newton = [newton(x, x_points, newton_coeff) for x in x_plot]
    newton_time = time.time() - start_time
    
    start_time = time.time()
    poly_coeff = get_coefficients(x_points, y_points)
    y_coeff = [eval_coefficients(x, poly_coeff) for x in x_plot]
    coeff_time = time.time() - start_time
    
    plt.plot(x_plot, y_true, 'k-', linewidth=2, label='Исходная функция', alpha=0.7)
    plt.plot(x_plot, y_lagrange, 'b--', linewidth=1.5, label='Лагранж', alpha=0.8)
    plt.plot(x_plot, y_newton, 'g-.', linewidth=1.5, label='Ньютон', alpha=0.8)
    plt.plot(x_plot, y_coeff, 'r:', linewidth=1.5, label='Коэффициенты (СЛАУ)', alpha=0.8)
    plt.plot(x_points, y_points, 'ro', markersize=6, label='Узлы интерполяции')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Интерполяция функции {name} (точек: {n_points}, шаг: {step})')
    plt.legend()
    
    print(f"Время Лагранжа: {lagrange_time:.6f} сек")
    print(f"Время Ньютона: {newton_time:.6f} сек")
    print(f"Время СЛАУ: {coeff_time:.6f} сек")
    
    if n_points > 10:
        max_diff_lag = np.max(np.abs(y_true - y_lagrange))
        max_diff_new = np.max(np.abs(y_true - y_newton))
        max_diff_coeff = np.max(np.abs(y_true - y_coeff))
        print(f"Макс. погрешность Лагранжа: {max_diff_lag:.6f}")
        print(f"Макс. погрешность Ньютона: {max_diff_new:.6f}")
        print(f"Макс. погрешность СЛАУ: {max_diff_coeff:.6f}")
    
    plt.show()

else:
    print("Неверный выбор. Запустите программу снова.")
