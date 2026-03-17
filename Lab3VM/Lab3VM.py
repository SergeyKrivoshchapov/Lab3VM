import matplotlib.pyplot as plt
import numpy as np

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

x_plot = np.linspace(-5, 15, 200)
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
