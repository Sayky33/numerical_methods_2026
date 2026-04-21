import numpy as np

def generate_and_save_data(n=100, x_val=2.5):
    A = np.random.rand(n, n) + np.eye(n) * n 
    
    X_true = np.full(n, x_val)
    
    B = np.dot(A, X_true)
    
    np.savetxt("matrix_A.txt", A, fmt='%.6f')
    np.savetxt("vector_B.txt", B, fmt='%.6f')
    print("Матриця A та вектор B згенеровані і збережені у файли.")
    
    return A, B, X_true

def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        U[i, i] = 1.0 
        
    for k in range(n):
        for i in range(k, n):
            sum_l = sum(L[i, j] * U[j, k] for j in range(k))
            L[i, k] = A[i, k] - sum_l
            
        for i in range(k + 1, n):
            sum_u = sum(L[k, j] * U[j, i] for j in range(k))
            U[k, i] = (A[k, i] - sum_u) / L[k, k]
            
    return L, U

def solve_lu(L, U, B):
    n = len(L)
    Z = np.zeros(n)
    X = np.zeros(n)
    
    for k in range(n):
        sum_z = sum(L[k, j] * Z[j] for j in range(k))
        Z[k] = (B[k] - sum_z) / L[k, k]
        
    for k in range(n - 1, -1, -1):
        sum_x = sum(U[k, j] * X[j] for j in range(k + 1, n))
        X[k] = Z[k] - sum_x
        
    return X

def vector_norm(V):
    return np.max(np.abs(V))

def iterative_refinement(A, B, L, U, X0, eps_0=1e-14):
    X = X0.copy()
    errors = []
    
    R = B - np.dot(A, X)
    current_eps = vector_norm(R)
    errors.append(current_eps)
    
    iteration = 0
    while current_eps > eps_0 and iteration < 50:
        delta_X = solve_lu(L, U, R) 
        
        X = X + delta_X
        
        R = B - np.dot(A, X)
        current_eps = vector_norm(R)
        errors.append(current_eps)
        
        iteration += 1
        
    return X, errors, iteration

if __name__ == "__main__":
    n_size = 100 
    eps_target = 1e-14 
    
    A, B, X_true = generate_and_save_data(n=n_size)
    L, U = lu_decomposition(A)
    
    np.savetxt("matrix_L.txt", L, fmt='%.6f')
    np.savetxt("matrix_U.txt", U, fmt='%.6f')
    
    X0 = solve_lu(L, U, B)
    
    initial_error = vector_norm(np.dot(A, X0) - B)
    print(f"Початкова норма нев'язки (eps): {initial_error:.2e}")
    X_refined, error_history, iters = iterative_refinement(A, B, L, U, X0, eps_0=eps_target)
    
    print(f"Кількість ітерацій: {iters}")
    print(f"Фінальна норма нев'язки: {error_history[-1]:.2e}")