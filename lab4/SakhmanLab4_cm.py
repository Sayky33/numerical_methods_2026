import numpy as np
import matplotlib.pyplot as plt

def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

t0 = 1.0  
exact_val = dM_exact(t0)
print(f"Точна похідна M'({t0}) = {exact_val:.10f}\n")

h_values = np.logspace(-20, 3, 500)

err_basic = np.zeros_like(h_values)
err_runge = np.zeros_like(h_values)
err_eitken = np.zeros_like(h_values)

best_h = h_values[0]
min_error = float('inf')

for i, h in enumerate(h_values):
    dh = central_diff(M, t0, h)
    d2h = central_diff(M, t0, 2 * h)
    d4h = central_diff(M, t0, 4 * h)

    err_basic[i] = abs(dh - exact_val)
    if err_basic[i] < min_error:
        min_error = err_basic[i]
        best_h = h
        
    y_runge = dh + (dh - d2h) / 3
    err_runge[i] = abs(y_runge - exact_val)
    
    term_bot = 2 * d2h - (d4h + dh)
    if term_bot != 0:
        y_eitken = (d2h**2 - d4h * dh) / term_bot
    else:
        y_eitken = dh
    err_eitken[i] = abs(y_eitken - exact_val)

print(f"Найкраща точність базового методу при h0 = {best_h:.7e}") 
print(f"Мінімальна похибка R0 = {min_error:.7e}\n") 

h_base = 1e-3
dh_base = central_diff(M, t0, h_base)
d2h_base = central_diff(M, t0, 2 * h_base)
d4h_base = central_diff(M, t0, 4 * h_base)
   
R1 = abs(dh_base - exact_val)
print(f"h = {h_base}: y'(h) = {dh_base:.10f}")
print(f"Похибка R1 = {R1:.2e}\n")

y_runge_base = dh_base + (dh_base - d2h_base) / 3
R2 = abs(y_runge_base - exact_val) 
print(f"Уточнене значення y'_R = {y_runge_base:.10f}")
print(f"Похибка R2 = {R2:.2e}\n")
    
term_bot_base = 2 * d2h_base - (d4h_base + dh_base)
y_eitken_base = (d2h_base**2 - d4h_base * dh_base) / term_bot_base if term_bot_base != 0 else dh_base
p = (1 / np.log(2)) * np.log(abs((d4h_base - d2h_base) / (d2h_base - dh_base)))
R3 = abs(y_eitken_base - exact_val) 
    
print(f"Уточнене значення y'_E = {y_eitken_base:.10f}")
print(f"Похибка R3 = {R3:.2e}")
print(f"Оцінений порядок точності p = {p:.2f}\n")

t_values = np.linspace(0, 20, 500)
start_func = M(t_values)

plt.plot(t_values, start_func)
plt.xlabel("Значення t")
plt.ylabel("Значення M(t)")
plt.title("Початкова функція")
plt.legend()
plt.show()

plt.figure(figsize=(10, 7))

plt.loglog(h_values, err_basic, linewidth=1.5, label="Загальна похибка R(h)", zorder=1, alpha=0.7)
plt.loglog(h_values, err_runge, linewidth=1.5, label="Метод Рунге-Ромберга", zorder=1, alpha=0.7)
plt.loglog(h_values, err_eitken, linewidth=1.5, label="Метод Ейткена", zorder=1, alpha=0.7)

plt.axvline(best_h, color='red', linestyle='--', label=f"Оптимальний $h_0 \\approx {best_h:.1e}$", zorder=2)

plt.xlabel("Крок h")
plt.ylabel("Абсолютна похибка R")
plt.title("Залежність похибки від кроку h та ефективність методів уточнення")
plt.grid(True, which="both", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()