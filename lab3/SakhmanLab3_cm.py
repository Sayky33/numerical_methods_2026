import csv
import numpy as np
import matplotlib.pyplot as plt

# 1. Вхідні дані
def read_data(filename):
    month = []
    temp = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            month.append(float(row['Month']))
            temp.append(float(row['Temp']))

    return month, temp

x, y = read_data('data.csv')

# 2. Функції МНК
def form_matrix(x, m):
    x = np.array(x)
    A = np.zeros((m+1, m+1))
    for i in range(m+1):
        for j in range(m+1):
            A[i, j] = np.sum(x ** (i + j))
    return A

def form_vector(x, y, m):
    x = np.array(x)
    b = np.zeros(m+1)
    for i in range(m+1):
        b[i] = np.sum(y * (x**i))
    return b

def gauss_solve(A, b):
    n = len(b)
    # Створюємо копії, щоб не модифікувати оригінальні масиви
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    
    # Прямий хід з вибором головного елемента
    for k in range(n-1):
        max_row = k + np.argmax(np.abs(A[k:n, k]))
        # Поміняти рядки k та max_row місцями
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
            
        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]
            
    # Зворотній хід
    x_sol = np.zeros(n)
    for i in range(n-1, -1, -1):
        # Віднімаємо суму вже знайдених змінних
        sum_ax = np.sum(A[i, i+1:] * x_sol[i+1:])
        x_sol[i] = (b[i] - sum_ax) / A[i, i]
        
    return x_sol

def polynomial(x, coef):
    x = np.array(x)
    y_poly = np.zeros_like(x, dtype=float)
    for i in range(len(coef)):
        y_poly += coef[i] * (x**i)
    return y_poly

def variance(y_true, y_approx):
    return np.mean((y_true - y_approx)**2)

# 3. Вибір оптимального ступеня полінома
max_degree = 4
variances = []

for m in range(1, max_degree + 1):
    A = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    coef = gauss_solve(A, b_vec)
    y_approx = polynomial(x, coef)
    var = variance(y, y_approx)
    variances.append(var)

# Знаходимо індекс мінімального значення (індексація з 0, тому +1 для ступеня)
optimal_m = np.argmin(variances) + 1

# 4. Побудова апроксимації (з оптимальним m)
A_opt = form_matrix(x, optimal_m)
b_opt = form_vector(x, y, optimal_m)
coef_opt = gauss_solve(A_opt, b_opt)
y_approx_opt = polynomial(x, coef_opt)

# 5. Прогноз на наступні 3 місяці
x_future = np.array([25, 26, 27])
y_future = polynomial(x_future, coef_opt)

# 6. Похибка апроксимації
error = y - y_approx_opt

# 7. Вивід результатів
print(f"Дисперсії для різних ступенів (m=1..{max_degree}):")
for m, var in enumerate(variances, 1):
    print(f"  Ступінь {m}: дисперсія = {var:.4f}")

print(f"\nОптимальний ступінь полінома: {optimal_m}")
print(f"Коефіцієнти полінома: {coef_opt}")
print(f"Прогноз на наступні 3 місяці (25, 26, 27): {y_future}")

# Побудова графіків
plt.figure(figsize=(10, 8))

# Перший графік: фактичні дані, апроксимація і прогноз
plt.subplot(2, 1, 1)
plt.scatter(x, y, color='blue', label='Фактичні температури', zorder=5)
plt.plot(x, y_approx_opt, color='green', label=f'Апроксимація (m={optimal_m})')
plt.scatter(x_future, y_future, color='red', marker='x', s=100, label='Прогноз (25-27)', zorder=5)
plt.plot(np.concatenate([x[-1:], x_future]), 
         np.concatenate([y_approx_opt[-1:], y_future]), 
         color='red', linestyle='--') # З'єднуємо апроксимацію і прогноз пунктиром
plt.title('Апроксимація МНК та прогноз температур')
plt.xlabel('Місяці')
plt.ylabel('Температура')
plt.grid(True)
plt.legend()

# Другий графік: похибка апроксимації
plt.subplot(2, 1, 2)
plt.bar(x, error, color='orange', label='Похибка (фактичне - апроксимація)')
plt.axhline(0, color='black', linewidth=1)
plt.title('Похибка апроксимації')
plt.xlabel('Місяці')
plt.ylabel('Величина похибки')
plt.grid(True, axis='y')
plt.legend()

plt.tight_layout()
plt.show()