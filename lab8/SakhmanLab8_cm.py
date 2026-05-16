import numpy as np
import matplotlib.pyplot as plt
import cmath

def F(x): return x * np.cos(x) - 1
def dF(x): return np.cos(x) - x * np.sin(x)
def d2F(x): return -2 * np.sin(x) - x * np.cos(x)

def tabulate_and_plot(a, b, h):
    x_vals = np.arange(a, b + h, h)
    y_vals = F(x_vals)
    
    with open("tabulation.txt", "w") as f:
        for x, y in zip(x_vals, y_vals):
            f.write(f"{x:.4f}\t{y:.4f}\n")
            
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label="F(x) = x*cos(x) - 1")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid()
    plt.legend()
    plt.title("Табуляція функції")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.show()

eps = 1e-10

def check_stop(x_new, x_old):
    return abs(F(x_new)) < eps and abs(x_new - x_old) < eps

def method_simple_iteration(x0, tau=-0.1, max_iters=1000):
    x = x0
    iters = 0
    while iters < max_iters:
        iters += 1
        x_new = x + tau * F(x)
        if check_stop(x_new, x): break
        x = x_new
    return x_new, iters

def method_newton(x0, max_iters=1000):
    x = x0
    iters = 0
    while iters < max_iters:
        iters += 1
        dfx = dF(x)
        if dfx == 0: break
        x_new = x - F(x) / dfx
        if check_stop(x_new, x): break
        x = x_new
    return x_new, iters

def method_chebyshev(x0, max_iters=1000):
    x = x0
    iters = 0
    while iters < max_iters:
        iters += 1
        fx, dfx, d2fx = F(x), dF(x), d2F(x)
        if dfx == 0: break
        x_new = x - (fx / dfx) - (0.5 * (fx**2) * d2fx) / (dfx**3)
        if check_stop(x_new, x): break
        x = x_new
    return x_new, iters

def method_chord(x0, x1, max_iters=1000):
    iters = 0
    while iters < max_iters:
        iters += 1
        fx0, fx1 = F(x0), F(x1)
        if fx1 - fx0 == 0: break
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if check_stop(x_new, x1): break
        x0, x1 = x1, x_new
    return x_new, iters

def method_parabolas(x0, x1, x2, max_iters=1000):
    iters = 0
    while iters < max_iters:
        iters += 1
        f0, f1, f2 = F(x0), F(x1), F(x2)
        
        f10 = (f1 - f0) / (x1 - x0) if x1 != x0 else 0
        f21 = (f2 - f1) / (x2 - x1) if x2 != x1 else 0
        f210 = (f21 - f10) / (x2 - x0) if x2 != x0 else 0
        
        A = f210
        B = (x2 - x1) * f210 + f21
        C = f2
        
        D = B**2 - 4*A*C
        sqrt_D = cmath.sqrt(D)
        
        if A == 0:
            delta = -C / B if B != 0 else 0
        else:
            delta1 = (-B + sqrt_D) / (2*A)
            delta2 = (-B - sqrt_D) / (2*A)
            delta = delta1 if abs(delta1) < abs(delta2) else delta2
        
        x_new = x2 + delta
        
        if isinstance(x_new, complex) and abs(x_new.imag) < eps:
            x_new = x_new.real
            
        if check_stop(x_new, x2): break
        x0, x1, x2 = x1, x2, x_new.real if isinstance(x_new, complex) else x_new
        
    return x_new, iters

def method_inverse_interpolation(x0, x1, x2, max_iters=1000):
    iters = 0
    while iters < max_iters:
        iters += 1
        y0, y1, y2 = F(x0), F(x1), F(x2)
        
        denom0 = (y0 - y1) * (y0 - y2)
        denom1 = (y1 - y0) * (y1 - y2)
        denom2 = (y2 - y0) * (y2 - y1)
        
        if denom0 == 0 or denom1 == 0 or denom2 == 0: break
            
        term0 = (y1 * y2 / denom0) * x0
        term1 = (y0 * y2 / denom1) * x1
        term2 = (y0 * y1 / denom2) * x2
        
        x_new = term0 + term1 + term2
        
        if check_stop(x_new, x2): break
        x0, x1, x2 = x1, x2, x_new
        
    return x_new, iters


def read_coeffs(filename="poly_coeffs.txt"):
    with open(filename, "r") as f:
        return list(map(float, f.read().split()))

def newton_horner(coeffs, x0, eps=1e-10, max_iters=1000):
    x = x0
    iters = 0
    m = len(coeffs) - 1
    while iters < max_iters:
        iters += 1
        b = [coeffs[0]]
        for i in range(1, m + 1):
            b.append(coeffs[i] + x * b[-1])
        
        c = [b[0]]
        for i in range(1, m):
            c.append(b[i] + x * c[-1])
            
        b0 = b[-1]
        c1 = c[-1]
        
        if c1 == 0: break
        
        x_new = x - b0 / c1
        if abs(x_new - x) < eps and abs(b0) < eps: break
        x = x_new
    return x_new, iters

def lin_method(coeffs, p0, q0, eps=1e-6, max_iters=1000):
    p, q = p0, q0
    iters = 0
    m = len(coeffs) - 1
    while iters < max_iters:
        iters += 1
        b = np.zeros(m + 1)
        b[0] = coeffs[0]
        if m >= 1:
            b[1] = coeffs[1] - p * b[0]
            
        for i in range(2, m + 1):
            b[i] = coeffs[i] - p * b[i-1] - q * b[i-2]
            
        R0 = b[m]
        R1 = b[m-1]
        
        if m > 2:
            denom = b[m-2]**2
            if denom == 0: denom = 1e-12
            dp = (R1 * b[m-2] - R0 * b[m-3]) / denom
        else:
            dp = R1 / (b[m-2] + 1e-12)
            
        dq = R0 / (b[m-2] + 1e-12)
        
        p_new = p + dp
        q_new = q + dq
        
        if abs(p_new - p) < eps and abs(q_new - q) < eps: break
        p, q = p_new, q_new
        
    alpha = -p / 2
    beta = np.sqrt(max(q - alpha**2, 0))
    return complex(alpha, beta), complex(alpha, -beta), iters

def plot_algebraic(coeffs, a=-3, b=3, h=0.1):
    """
    Будує графік алгебраїчного полінома.
    coeffs - список коефіцієнтів від найстаршого степеня до вільного члена.
    """
    x_vals = np.arange(a, b + h, h)
    y_vals = np.polyval(coeffs, x_vals)
    
    plt.figure(figsize=(8, 5))
    
    m = len(coeffs) - 1
    poly_str = " + ".join([f"{c}x^{m-i}" if m-i > 0 else f"{c}" for i, c in enumerate(coeffs)])
    
    plt.plot(x_vals, y_vals, color='red', label=f"F(x) = {poly_str}")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid()
    plt.legend()
    plt.title("Графік алгебраїчного рівняння")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.show()


if __name__ == "__main__":
    print("Трансцендентні рівняння")
    tabulate_and_plot(-3, 6, 0.1)

    roots_guesses = [-2.0, 4.9]

    for guess in roots_guesses:
        print(f"Пошук кореня з початковим наближенням x0 = {guess}")
        
        safe_tau = -0.1 * np.sign(dF(guess))
        if safe_tau == 0: safe_tau = -0.1
        
        res_iter, i_iter   = method_simple_iteration(guess, tau=safe_tau)
        res_newton, i_newton = method_newton(guess)
        res_cheb, i_cheb   = method_chebyshev(guess)
        res_chord, i_chord  = method_chord(guess - 0.1, guess)
        
        res_parab, i_parab = method_parabolas(guess - 0.2, guess - 0.1, guess)
        res_inv, i_inv     = method_inverse_interpolation(guess - 0.2, guess - 0.1, guess)
        
        print(f"Проста ітерація:       x = {res_iter:13.10f}, ітерацій: {i_iter}")
        print(f"Метод Ньютона:         x = {res_newton:13.10f}, ітерацій: {i_newton}")
        print(f"Метод Чебишева:        x = {res_cheb:13.10f}, ітерацій: {i_cheb}")
        print(f"Метод хорд:            x = {res_chord:13.10f}, ітерацій: {i_chord}")
        print(f"Метод парабол:         x = {float(res_parab.real):13.10f}, ітерацій: {i_parab}")
        print(f"Зворотна інтерполяція: x = {res_inv:13.10f}, ітерацій: {i_inv}")
    
    print(f"\nАлгебраїчні рівняння")
    
    A = [1.0, -1.0, 0.0, 2.0] 
    with open("poly_coeffs.txt", "w") as f:
        f.write(" ".join(map(str, A)))

    coeffs = read_coeffs()
    print(f"Зчитані коефіцієнти рівняння: {coeffs}")

    plot_algebraic(coeffs, a=-2, b=3)

    real_root, iters_h = newton_horner(coeffs, -1.5)
    print(f"Дійсний корінь (Ньютон-Горнер): x  = {real_root:.10f}, ітерацій: {iters_h}")

    comp1, comp2, iters_l = lin_method(coeffs, p0=-1.5, q0=1.5)
    print(f"Комплексні корені (Метод Ліна): x1 = {comp1:.10f}")
    print(f"                                x2 = {comp2:.10f}, ітерацій: {iters_l}")