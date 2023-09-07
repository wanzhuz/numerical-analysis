import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time


def f(x):
    f1 = x[0] ** 2 - 10 * x[0] + x[1] ** 2 + 8
    f2 = x[0] * x[1] ** 2 + x[0] - 10 * x[1] + 8
    return np.array([f1, f2])


root = fsolve(f, [1, 1])


def g(x):
    g1 = (x[0] ** 2 + x[1] ** 2 + 8) / 10.0
    g2 = (x[0] * x[1] ** 2 + x[1] + 8) / 10.0
    return np.array([g1, g2])


def fixed_point(x_old, tol=1e-6, max_iter=1000):
    step = 1
    norm = np.array([])
    for i in range(max_iter):
        x_new = g(x_old)
        x_old = x_new
        step += 1
        norm = np.append(norm, np.linalg.norm(x_new - root, ord=np.inf))
        if np.linalg.norm(x_new - root, ord=np.inf) < tol:
            break
    return x_new, step, norm


# print(fixed_point(np.array([0, 0])))


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


# backwards substitution
def back_sub(A, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x


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


def jacobian(x):
    J = np.array([[2 * x[0] - 10, 2 * x[1]],
                  [x[1] ** 2 + 1, 2 * x[0] * x[1] - 10]])
    return J


def newton(x, tol=1e-6, max_iter=1000):
    step = 1
    norm = np.array([])
    for k in range(max_iter):
        y = lu_solver(jacobian(x), -f(x))
        x = x + y
        step += 1
        norm = np.append(norm, np.linalg.norm(x - root, ord=np.inf))
        if np.linalg.norm(x - root, ord=np.inf) < tol:
            break
    return x, step, norm


# print(newton(np.array([0, 0])))


def clock(type=0):
    if type == 0:
        t0 = time.time()
        fixed_point(np.array([0, 0]))
        t1 = time.time()
        total_time = t1 - t0
        return total_time
    if type == 1:
        t0 = time.time()
        newton(np.array([0, 0]))
        t1 = time.time()
        total_time = t1 - t0
        return total_time


# print(clock(type=0))
# print(clock(type=1))

iter1 = np.linspace(1, 15, num=15)
iter2 = np.linspace(1, 4, num=4)
n_error = newton(np.array([0, 0]))[2]
newton_error = n_error.tolist()
f_error = fixed_point(np.array([0, 0]))[2]
fixed_error = f_error.tolist()

plt.plot(iter1, fixed_error)
# plt.plot(iter2, newton_error)
plt.xlabel("Iterations")
plt.ylabel("Eigenvalue Error")
plt.title("Eigenvalue Error at each Iteration")
# plt.legend(["Fixed Point Method", "Newton's Method"], loc="upper right")
plt.show()
