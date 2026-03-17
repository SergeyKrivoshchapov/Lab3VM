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

x_points = [5, 7, 9, 11, 12]
y_points = [3, -2, -2, 4, 15]

for i in range(len(x_points)):
    x = x_points[i]
    y = lagrange(x, x_points, y_points)

test_points = [8, 11, 13]
for x in test_points:
    y = lagrange(x, x_points, y_points)
    print(f"L({x}) = {y:.6f}")

import matplotlib.pyplot as plt
import numpy as np

x_plot = np.linspace(6, 15, 200)
y_plot = [lagrange(x, x_points, y_points) for x in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Многочлен Лагранжа')
plt.plot(x_points, y_points, 'ro', markersize=8, label='Узлы интерполяции')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('x')
plt.ylabel('L(x)')
plt.legend()

for i, (x, y) in enumerate(zip(x_points, y_points)):
    plt.annotate(f'({x}, {y})', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.show()
