import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24

x = np.linspace(0, 24, 1000)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$')
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True)
plt.legend()
plt.show()

I0, _ = quad(f, a, b, epsabs=1e-15, epsrel=1e-15)

def simpson_composite(f, a, b, N):
    if N % 2 != 0: 
        N += 1 
    
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    
    S = y[0] + y[-1]
    for i in range(1, N):
        if i % 2 == 0:
            S += 2 * y[i]
        else:
            S += 4 * y[i] 
            
    return (h / 3) * S

N_vals = np.arange(10, 1002, 2)
eps_vals = [abs(simpson_composite(f, a, b, n) - I0) for n in N_vals]

N_opt = 10
target_eps = 1e-12
for n in range(10, 5000, 2):
    if abs(simpson_composite(f, a, b, n) - I0) <= target_eps:
        N_opt = n
        break
eps_opt = abs(simpson_composite(f, a, b, N_opt) - I0)

plt.figure(figsize=(10, 5))
plt.semilogy(N_vals, eps_vals, label="|I(N) - I_0|", color='blue')
plt.axhline(target_eps, color='red', linestyle='--', label=f"Задана точність {target_eps}")
plt.xlabel("Число розбиттів (N)")
plt.ylabel("Абсолютна похибка (логарифмічна шкала)")
plt.title("Залежність похибки формули Сімпсона від N")
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend()
plt.show()

N0 = max(8, round((N_opt / 10) / 8) * 8)
I_N0 = simpson_composite(f, a, b, N0)
eps0 = abs(I_N0 - I0)

I_N0_2 = simpson_composite(f, a, b, N0 // 2)
I_R = I_N0 + (I_N0 - I_N0_2) / 15
epsR = abs(I_R - I0)

I_N0_4 = simpson_composite(f, a, b, N0 // 4)

den_aitken = 2 * I_N0_2 - (I_N0 + I_N0_4)
if abs(den_aitken) > 1e-16:
    I_E = (I_N0_2**2 - I_N0 * I_N0_4) / den_aitken
else:
    I_E = I_N0 
epsE = abs(I_E - I0)

num_p = I_N0_4 - I_N0_2
den_p = I_N0_2 - I_N0
if abs(den_p) > 1e-16 and abs(num_p / den_p) > 0:
    p_est = (1 / np.log(2)) * np.log(abs(num_p / den_p))
else:
    p_est = float('nan')

def adaptive_simpson(f, a, b, delta, depth=0, max_depth=50):
    h = b - a
    c = (a + b) / 2
    
    I1 = (h / 6) * (f(a) + 4*f(c) + f(b))
    
    d = (a + c) / 2
    e = (c + b) / 2
    I2 = (h / 12) * (f(a) + 4*f(d) + 2*f(c) + 4*f(e) + f(b))
    
    if depth >= max_depth or abs(I1 - I2) <= delta:
        return I2, 5
    else:
        left_val, left_calls = adaptive_simpson(f, a, c, delta / 2, depth + 1, max_depth)
        right_val, right_calls = adaptive_simpson(f, c, b, delta / 2, depth + 1, max_depth)
        return left_val + right_val, left_calls + right_calls + 5

print("-" * 50)
print(f"Точне значення I0 = {I0:.12f}")
print(f"N_opt (для eps=1e-12) = {N_opt}, eps_opt = {eps_opt:.2e}")
print("-" * 50)
print(f"Базове обчислення (N0 = {N0}):")
print(f"I(N0)   = {I_N0:.12f}, eps0 = {eps0:.2e}")
print("-" * 50)
print("Методи підвищення точності:")
print(f"Рунге-Ромберг: I_R = {I_R:.12f}, epsR = {epsR:.2e}")
print(f"Ейткен:        I_E = {I_E:.12f}, epsE = {epsE:.2e}")
print(f"Оцінений порядок методу (Ейткен) p = {p_est:.4f} (Теоретичний: 4.0)")
print("-" * 50)

print("Дослідження адаптивного алгоритму:")
print(f"{'Допуск':<15} | {'Значення інтегралу':<18} | {'Похибка':<12} | {'Виклики f(x)':<12}")
print("-" * 65)
for d_val in [1e-2, 1e-4, 1e-6, 1e-9, 1e-12]:
    val, calls = adaptive_simpson(f, a, b, d_val)
    print(f"{d_val:<15.0e} | {val:<18.12f} | {abs(val - I0):<12.2e} | {calls:<12}")