import numpy as np

x = [5, 7, 9, 11, 12]
y = [3, -2, -2, 4, 15]

# Пример системы:
# 2x + 3y = 5
# 4x -  y = 1

A = np.array([[1, x[0], x[0]**2, x[0]**3, x[0]**4],
              [1, x[1], x[1]**2, x[1]**3, x[1]**4],
              [1, x[2], x[2]**2, x[2]**3, x[2]**4],
              [1, x[3], x[3]**2, x[3]**3, x[3]**4],
              [1, x[4], x[4]**2, x[4]**3, x[4]**4]
              ], dtype=float)   # матрица коэффициентов
b = np.array([y[0], y[1], y[2], y[3], y[4]], dtype=float)      # вектор свободных членов

a = np.linalg.solve(A, b)
print(a)

import matplotlib.pyplot as plt

x = np.linspace(-5, 12, 100)  # Диапазон x от -10 до 10
y = a[0]*x**0 + a[1] * x**1 + a[2] * x**2 + a[3] * x**3 + a[4] * x**4

plt.plot(x, y)  # Строим линейный график
plt.xlabel('x')  # Подпись оси X
plt.ylabel('y')  # Подпись оси 
plt.xlim(4, 13)
plt.ylim(-3, 16)

plt.title('График y = x²') 
plt.grid(True) 
plt.show()  