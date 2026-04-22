import numpy as np

def generate_and_save_data(n=100, exact_val=2.5, matrix_file='matrix_A.txt', vector_file='vector_b.txt'):
    A = np.random.rand(n, n) * 10
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + np.random.rand() * 10 

    np.savetxt(matrix_file, A, fmt='%.6f')

    X_exact = np.full(n, exact_val)
    
    b = np.dot(A, X_exact)
    
    np.savetxt(vector_file, b, fmt='%.6f')
    
    return A, b

def load_data(matrix_file='matrix_A.txt', vector_file='vector_b.txt'):
    A = np.loadtxt(matrix_file)
    b = np.loadtxt(vector_file)
    return A, b

def vector_norm(v):
    return np.max(np.abs(v)) 

def matrix_norm(A):
    return np.max(np.sum(np.abs(A), axis=1)) 

def simple_iteration_method(A, b, x0, epsilon=1e-14, max_iter=10000):
    n = len(b)
    x = np.copy(x0)
    
    tau = 1.0 / matrix_norm(A) 
    
    E = np.eye(n)
    C = E - tau * A
    d = tau * b
    
    for k in range(max_iter):
        x_new = np.dot(C, x) + d
        if vector_norm(x_new - x) < epsilon:
            return x_new, k + 1
        x = x_new
    return x, max_iter

def jacobi_method(A, b, x0, epsilon=1e-14, max_iter=10000):
    n = len(b)
    x = np.copy(x0)
    x_new = np.zeros_like(x)
    
    for k in range(max_iter):
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
            
        if vector_norm(x_new - x) < epsilon:
            return x_new, k + 1
        x = np.copy(x_new)
    return x, max_iter

def seidel_method(A, b, x0, epsilon=1e-14, max_iter=10000):
    n = len(b)
    x = np.copy(x0)
    
    for k in range(max_iter):
        x_old = np.copy(x)
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i)) 
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i, i]
            
        if vector_norm(x - x_old) < epsilon:
            return x, k + 1
    return x, max_iter

if __name__ == "__main__":
    n_size = 100
    eps = 1e-14
    
    generate_and_save_data(n=n_size)
    
    A, b = load_data()
    
    x0 = np.ones(n_size)
    
    print(f"\nТочність = {eps}")
    
    x_simple, iter_simple = simple_iteration_method(A, b, x0, epsilon=eps)
    print(f"Метод простої ітерації: {iter_simple} ітерацій. Середнє значення X: {np.mean(x_simple):.6f}")
    
    x_jacobi, iter_jacobi = jacobi_method(A, b, x0, epsilon=eps)
    print(f"Метод Якобі:           {iter_jacobi} ітерацій. Середнє значення X: {np.mean(x_jacobi):.6f}")
    
    x_seidel, iter_seidel = seidel_method(A, b, x0, epsilon=eps)
    print(f"Метод Зейделя:         {iter_seidel} ітерацій. Середнє значення X: {np.mean(x_seidel):.6f}")