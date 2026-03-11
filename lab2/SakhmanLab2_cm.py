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

    def w_seeker(some_x, k):
        res = 1
        for i in range(k + 1):
            res *= (some_x - x[i])
        return res
    
    def divided_diff_seeker(k):
        total = 0
        for i in range(k + 1):
            denom = 1
            for j in range(k + 1):
                if j != i:
                    denom *= (x[i] - x[j])
            
            total += y[i]/denom
        return total
    
    def interpolation(some_x):
        mid_sum = 0
        for i in range(1, n):
            mid_sum += w_seeker(some_x, i - 1) * divided_diff_seeker(i)
        return mid_sum + y[0]

    def error(some_x):
        ind = x.index(some_x)
        return abs(y[ind] - interpolation(some_x))
    
    predicted = []
    # some_error = []
    w = []
    for prediction in x_predict:
        predicted.append(interpolation(prediction))
        # some_error.append(error(prediction))
        w.append(w_seeker(prediction, n - 1))

    return predicted, w#, some_error



x, y = read_data("data.csv")
print("Tabulation:")
print("| Datasets | Time(sec) |")
for i in range(len(x)):
    print(f"| {x[i]:8} | {y[i]:9} |")

v = [10000, 20000, 40000, 80000, 120000, 160000]

h = (160000 - 10000)/100
f = [10000]
p = 10000
while p <= 160000:
    p += h
    f.append(p)
# a, b = Newtons_interpolation(x, y, v)
# print(a, b)
a, b = Newtons_interpolation(x, y, f)
f = np.array(f)
a = np.array(a)
b = np.array(b)

plt.figure()
plt.plot(f, b)
plt.plot(f, a)

plt.show()