import time
import numpy as np
import matplotlib.pyplot as plt


# gaussian elimination
def gaussian(A, b):
    n = len(b)
    B = np.copy(A)
    p = np.copy(b)
    for i in range(n-1):
        for j in range(i+1, n):
            r = B[j][i] / B[i][i]
            for k in range(i, n):
                B[j][k] -= r * B[i][k]
            p[j] -= r * p[i]
    return back_sub(B, p)


# backwards substitution
def back_sub(A, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x


# LU decomposition
def lu(A):
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    N = np.size(A, 0)

    for k in range(N):
        L[k, k] = 1
        U[k, k] = (A[k, k] - np.dot(L[k, :k], U[:k, k])) / L[k, k]
        for j in range(k + 1, N):
            U[k, j] = (A[k, j] - np.dot(L[k, :k], U[:k, j])) / L[k, k]
        for i in range(k + 1, N):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]
    return L, U


# forward substitution
def forward_sub(A, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i] - np.dot(A[i, :i], x[:i]) / A[i][i]
    return x


# LUx = b
def lu_solver(A, b):
    L = lu(A)[0]
    U = lu(A)[1]
    y = forward_sub(L, b)
    x = back_sub(U, y)
    return x


# Q3
A = np.array([[2, 4, 5], [7, 6, 5], [9, 11, 3]], dtype=float)
b = np.array([3, 2, 1], dtype=float)

print(gaussian(A, b))
print(lu(A))
print(lu_solver(A, b))


def mat(n):
    A = 5 * np.sqrt(n) * np.eye(n) + np.random.normal(size=(n, n))
    return A


def sol(n):
    b = np.random.normal(size=(n, 1))
    return b


# GE
print(gaussian(mat(50), sol(50)))
print(gaussian(mat(100), sol(100)))
print(gaussian(mat(150), sol(250)))
print(gaussian(mat(200), sol(500)))

# LU
print(lu_solver(mat(50), sol(50)))
print(lu_solver(mat(100), sol(100)))
print(lu_solver(mat(150), sol(250)))
print(lu_solver(mat(200), sol(500)))


# residual errors
def ge_residual(n):
    A = mat(n)
    b = np.squeeze(sol(n))
    x = gaussian(A, b)
    P = np.dot(A, x)
    residual = np.linalg.norm(P - b, 2)
    return residual


def lu_residual(n):
    A = mat(n)
    b = np.squeeze(sol(n))
    x = lu_solver(A, b)
    P = np.dot(A, x)
    residual = np.linalg.norm(P - b, 2)
    return residual


print(ge_residual(50))
print(lu_residual(50))
print(ge_residual(100))
print(lu_residual(100))
print(ge_residual(250))
print(lu_residual(250))
print(ge_residual(500))
print(lu_residual(500))

# wall clock time
def clock(n, type=0):
    A = mat(n)
    b = sol(n)
    if type == 0:
        t0 = time.time()
        gaussian(A, b)
        t1 = time.time()
        total_time = t1 - t0
        return total_time
    if type == 1:
        t0 = time.time()
        lu_solver(A, b)
        t1 = time.time()
        total_time = t1 - t0
        return total_time


print(clock(50, 0))
print(clock(100, 0))
print(clock(50, 1))
print(clock(100, 0))
print(clock(100, 1))
print(clock(250, 0))
print(clock(250, 1))
print(clock(500, 0))
print(clock(500, 1))


# plotting
times_ge = np.empty((0, 0))
times_ge = np.append(times_ge, (clock(50, 0), clock(100, 0), clock(250, 0), clock(500, 0)))
times_lu = np.empty((0, 0))
times_lu = np.append(times_lu, (clock(50, 1), clock(100, 1), clock(250, 1), clock(500, 1)))

dims = [50, 100, 250, 500]
plt.plot(dims, times_ge)
plt.plot(dims, times_lu)
plt.xlabel("n")
plt.ylabel("Time Elapsed (seconds)")
plt.title("Execution Time of Solving Linear Systems")
plt.legend(["Gaussian Elimination", "LU Decomposition"], loc="upper left")
plt.show()

