import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x - y

def exact_sol(x):
    return x - 1 + 2 * np.exp(-x)

def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h, y + h*k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def adaptive_rk4(f, x0, y0, x_end, tol):
    x, y, h_arr, est_err, true_err = [x0], [y0], [], [0], [0]
    h = 0.01
    
    while x[-1] < x_end:
        y_full = rk4_step(f, x[-1], y[-1], h)
        y_half1 = rk4_step(f, x[-1], y[-1], h/2)
        y_half2 = rk4_step(f, x[-1] + h/2, y_half1, h/2)
        
        err = (16/15) * abs(y_half2 - y_full)
        
        if err > tol:
            h /= 2 
            continue
            
        x.append(x[-1] + h)
        y.append(y_half2)
        h_arr.append(h)
        est_err.append(err)
        true_err.append(abs(y[-1] - exact_sol(x[-1])))
        
        if err < tol / 32:
            h *= 2
            
    return np.array(x), np.array(y), np.array(h_arr), np.array(est_err), np.array(true_err)

def adaptive_adams2(f, x0, y0, x_end, tol):
    x, y, h_arr, est_err, true_err = [x0], [y0], [], [0], [0]
    h = 0.01

    x.append(x0 + h)
    y.append(rk4_step(f, x0, y0, h))
    h_arr.append(h)
    est_err.append(0)
    true_err.append(abs(y[-1] - exact_sol(x[-1])))
    
    while x[-1] < x_end:
        xn, yn = x[-1], y[-1]
        x_prev, y_prev = x[-2], y[-2]
        
        fn = f(xn, yn)
        fn_prev = f(x_prev, y_prev)
        
        y_pre = yn + (h/2) * (3*fn - fn_prev)
        
        y_cor = yn + (h/2) * (f(xn + h, y_pre) + fn)
        
        err = abs(y_cor - y_pre)
        
        if err > tol:
            h /= 2
            x.pop()
            y.pop()
            x.append(x[-1] + h)
            y.append(rk4_step(f, x[-1], y[-1], h))
            continue
            
        x.append(xn + h)
        y.append(y_cor)
        h_arr.append(h)
        est_err.append(err)
        true_err.append(abs(y[-1] - exact_sol(x[-1])))
        
        if err < tol / 10:
            h *= 2
            
    return np.array(x), np.array(y), np.array(h_arr), np.array(est_err), np.array(true_err)

tol_rk = 5e-2
tol_adams = 5e-2
x_end = 2.0

x_rk, y_rk, h_rk, err_est_rk, err_true_rk = adaptive_rk4(f, 0, 1, x_end, tol_rk)
x_ad, y_ad, h_ad, err_est_ad, err_true_ad = adaptive_adams2(f, 0, 1, x_end, tol_adams)

fig, axs = plt.subplots(3, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.4)

axs[0, 0].plot(x_ad, err_true_ad, 'b.-', markersize=4, label='Істинна похибка $\phi_n$')
axs[0, 0].set_title('Метод Адамса: Локальна похибка')
axs[0, 0].grid(True)
axs[0, 0].legend()

axs[0, 1].plot(x_ad, err_est_ad, 'r.-', markersize=4, label='$|y^{cor} - y^{pre}|$')
axs[0, 1].set_title('Метод Адамса: Оцінка похибки')
axs[0, 1].grid(True)
axs[0, 1].legend()

axs[1, 0].step(x_ad[:-1], h_ad, 'g.-', where='post', markersize=4, label='Крок $h(x)$')
axs[1, 0].set_title('Метод Адамса: Адаптивний крок')
axs[1, 0].grid(True)
axs[1, 0].legend()

axs[1, 1].plot(x_rk, err_true_rk, 'b.-', markersize=4, label='Істинна похибка $\phi_n$')
axs[1, 1].set_title('Метод Рунге-Кутта: Локальна похибка')
axs[1, 1].grid(True)
axs[1, 1].legend()

axs[2, 0].plot(x_rk, err_est_rk, 'm.-', markersize=4, label='Похибка Рунге')
axs[2, 0].set_title('Метод Рунге-Кутта: Оцінка похибки')
axs[2, 0].grid(True)
axs[2, 0].legend()

axs[2, 1].step(x_rk[:-1], h_rk, 'g.-', where='post', markersize=4, label='Крок $h(x)$')
axs[2, 1].set_title('Метод Рунге-Кутта: Адаптивний крок')
axs[2, 1].grid(True)
axs[2, 1].legend()

for ax in axs.flat:
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

plt.show()