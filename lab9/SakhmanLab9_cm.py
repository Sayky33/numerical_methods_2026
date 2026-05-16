import numpy as np
import matplotlib.pyplot as plt

def hooke_jeeves(func, x0, delta_x, q=2.0, p=2.0, eps1=1e-5, eps2=1e-5, max_iter=1000):
    """
    Метод Хука-Дживса багатовимірної оптимізації.
    """
    x_base = np.array(x0, dtype=float)
    dx = np.array(delta_x, dtype=float)
    
    n = len(x_base)
    trajectory = [x_base.copy()]
    steps_count = 0
    
    def explore(x_start, current_dx, allow_reduce=True):
        x_new = x_start.copy()
        dx_temp = current_dx.copy()
        
        for i in range(n):
            while True:
                f_base = func(x_start)
                
                x_temp = x_new.copy()
                x_temp[i] += dx_temp[i]
                if func(x_temp) < f_base:
                    x_new = x_temp
                    break
                
                x_temp[i] = x_new[i] - 2 * dx_temp[i]
                if func(x_temp) < f_base:
                    x_new = x_temp
                    break
                
                if allow_reduce:
                    dx_temp[i] /= q
                    if dx_temp[i] < eps1:
                        break
                else:
                    break
        return x_new, dx_temp

    for _ in range(max_iter):
        steps_count += 1
        
        x1, dx = explore(x_base, dx, allow_reduce=True)
        
        if np.array_equal(x1, x_base) or (np.linalg.norm(dx) < eps1 and abs(func(x1) - func(x_base)) < eps2):
            break
            
        trajectory.append(x1.copy())
        
        while True:
            xp = x1 + p * (x1 - x_base)
            
            x2, _ = explore(xp, dx, allow_reduce=False)
            
            if func(x2) < func(x1):
                x_base = x1.copy()
                x1 = x2.copy()
                trajectory.append(x1.copy())
            else:
                x_base = x1.copy()
                break

    return x1, trajectory, steps_count

def rosenbrock(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

x0_rosenbrock = [-1.2, 0.0]
delta_x_init = [0.5, 0.5]

res_x, traj, steps = hooke_jeeves(rosenbrock, x0_rosenbrock, delta_x_init)
print("Функція Розенброка")
print(f"Знайдений мінімум: {res_x}")
print(f"Значення функції: {rosenbrock(res_x):.6f}")
print(f"Кількість кроків: {steps}\n")

def target_function(x):
    f1 = x[0]**2 + x[1]**2 - 4
    f2 = x[0] - x[1]
    return f1**2 + f2**2

x0_system = [2.0, 1.0] 
delta_x_sys = [0.5, 0.5]

res_sys, traj_sys, steps_sys = hooke_jeeves(target_function, x0_system, delta_x_sys)
print("Розв'язок системи рівнянь")
print(f"Знайдений корінь: x1 = {res_sys[0]:.5f}, x2 = {res_sys[1]:.5f}")
print(f"Похибка (значення цільової функції): {target_function(res_sys):.2e}")
print(f"Кількість кроків на траєкторії спуску: {steps_sys}") # 

with open("trajectory_output.txt", "w") as f:
    f.write("Траєкторія спуску (x1, x2):\n")
    for i, point in enumerate(traj_sys):
        f.write(f"Крок {i}: ({point[0]:.5f}, {point[1]:.5f})\n")
x1 = np.linspace(-2.5, 2.5, 400)

x2 = x1**2

plt.figure(figsize=(12, 6))

plt.plot(x1, x2, 'b-', label=r'$10(x_1^2 - x_2) = 0$', linewidth=1.5)

plt.axvline(x=1, color='r', label=r'$x_1 - 1 = 0$', linewidth=1.5)

plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

plt.xlim(-2.0, 2.0)
plt.ylim(-1.0, 3.0)

plt.grid(True, linestyle='--', alpha=0.7)
plt.title("Графіки рівнянь для функції Розенброка")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(loc='upper right')

plt.show()