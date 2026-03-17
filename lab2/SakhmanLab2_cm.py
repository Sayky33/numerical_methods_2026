import csv
import matplotlib.pyplot as plt
import numpy as np

def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return x, y

def Newtons_interpolation(x, y, x_predict):
    n = len(x)
    
    coef = np.zeros((n, n))
    
    for i in range(n):
        coef[i][0] = y[i]
        
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
            

    def interpolation(some_x):
        res = coef[0, 0]
        w = 1.0
        for i in range(1, n):
            w *= (some_x - x[i - 1])
            res += coef[0, i] * w
        return res
    
    predicted = []
    for prediction in x_predict:
        predicted.append(interpolation(prediction))

    return predicted
def Factorial_interpolation(x, y, predict):
    n = len(x)
    h = x[1] - x[0] 

    def fdiff_seeker():
        diffs = [[0] * n for _ in range(n)]
        for i in range(n):
            diffs[i][0] = y[i]
            
        for j in range(1, n):
            for i in range(n - j):
                diffs[i][j] = diffs[i+1][j-1] - diffs[i][j-1]
                
        return diffs[0]
    
    delta_f0 = fdiff_seeker()

    def t_factorial(t, k):
        res = 1
        for i in range(k):
            res *= (t - i)
        return res
    
    def factorial(num):
        if num == 0: 
            return 1
        res = 1
        for i in range(1, num + 1):
            res *= i
        return res

    def interpolation(some_x):
        t = (some_x - x[0]) / h 
        
        result = 0
        for k in range(n):
            term = (delta_f0[k] / factorial(k)) * t_factorial(t, k)
            result += term
            
        return result

    predicted = []
    
    for prediction in predict:
        predicted.append(interpolation(prediction))

    return predicted

x, y = read_data("data.csv")
print("Tabulation:")
print("| Datasets | Time(sec) |")
for i in range(len(x)):
    print(f"| {x[i]:8} | {y[i]:9} |")

a = x[0]
b = x[-1]

ab_arr = list(np.linspace(a, b, 6)) 
graph = list(np.linspace(a, b, 100))
need = [120000]

newt_nodes = Newtons_interpolation(x, y, ab_arr)

pred = Newtons_interpolation(x, y, need)
predf = Factorial_interpolation(ab_arr, newt_nodes, need)

print(f'\nPredicted({need[0]}) = {pred[0]:.2f}')
print(f'Predictedf({need[0]}) = {predf[0]:.2f}\n')

newt_graph = Newtons_interpolation(x, y, graph)
fact_graph = Factorial_interpolation(ab_arr, newt_nodes, graph)

err = np.abs(np.array(fact_graph) - np.array(newt_graph))

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Exp. points (Original)', zorder=5)
plt.plot(graph, newt_graph, label='Newtons plot', color='blue')
plt.plot(graph, fact_graph, label='Factorials plot', color='orange', linestyle='--') 

plt.scatter(need, pred, color='cyan', s=80, label='Newton Predict', zorder=5)
plt.scatter(need, predf, marker='x', s=80, label='Factorial Predict', zorder=6)

plt.title("Interpolation comparison")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(graph, err, label='Error |Factorial - Newton|', color='purple')
plt.title("Absolute error")
plt.legend()
plt.grid(True)
plt.show()


nodes5_x = list(np.linspace(a, b, 5))
nodes5_y = Newtons_interpolation(x, y, nodes5_x)

nodes10_x = list(np.linspace(a, b, 10))
nodes10_y = Newtons_interpolation(x, y, nodes10_x)

nodes20_x = list(np.linspace(a, b, 20))
nodes20_y = Newtons_interpolation(x, y, nodes20_x)

pred_newt5 = np.array(Newtons_interpolation(nodes5_x, nodes5_y, graph))
pred_newt10 = np.array(Newtons_interpolation(nodes10_x, nodes10_y, graph))
pred_newt20 = np.array(Newtons_interpolation(nodes20_x, nodes20_y, graph))

pred_fact5 = np.array(Factorial_interpolation(nodes5_x, nodes5_y, graph))
pred_fact10 = np.array(Factorial_interpolation(nodes10_x, nodes10_y, graph))
pred_fact20 = np.array(Factorial_interpolation(nodes20_x, nodes20_y, graph))

true_y = np.array(newt_graph)

errors_newt5 = np.abs(true_y - pred_newt5)
errors_newt10 = np.abs(true_y - pred_newt10)
errors_newt20 = np.abs(true_y - pred_newt20)

errors_fact5 = np.abs(true_y - pred_fact5)
errors_fact10 = np.abs(true_y - pred_fact10)
errors_fact20 = np.abs(true_y - pred_fact20)

fig, ax = plt.subplots(figsize=(10, 4))

ax.grid(linestyle='-', linewidth=0.5, alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=8)

ax.plot(graph, errors_newt5, label='|Похибка| 5 вузлів (Ньютон)', linestyle='-')
ax.plot(graph, errors_newt10, label='|Похибка| 10 вузлів (Ньютон)', linestyle='-')
ax.plot(graph, errors_newt20, label='|Похибка| 20 вузлів (Ньютон)', linestyle='-')

ax.plot(graph, errors_fact5, label='|Похибка| 5 вузлів (Факт.)', linestyle='--')
ax.plot(graph, errors_fact10, label='|Похибка| 10 вузлів (Факт.)', linestyle='--')
ax.plot(graph, errors_fact20, label='|Похибка| 20 вузлів (Факт.)', linestyle='--')

ax.set_xlabel('n', fontsize=10)
ax.set_ylabel('Абсолютна похибка', fontsize=10)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False, fontsize=9)

plt.tight_layout()
plt.show()