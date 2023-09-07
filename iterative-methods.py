import numpy as np
import matplotlib.pyplot as plt
import time


def jacobi(A, b, tol=1e-8, max_iter=1000):
    x = np.zeros_like(b, dtype=float)
    T = A - np.diag(np.diagonal(A))

    for k in range(max_iter):
        x_og = x.copy()
        x[:] = (b - np.dot(T, x)) / np.diagonal(A)
        if np.linalg.norm(x - x_og, ord=np.inf) < tol:
            break
    return x, k +1


def gauss_seidel(A, b, tol=1e-8, max_iter=1000):
    x = np.zeros_like(b, dtype=float)

    for k in range(max_iter):
        x_og = x.copy()
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, (i + 1):], x_og[(i + 1):])) / A[i, i]
        if np.linalg.norm(x - x_og, ord=np.inf) < tol:
            break
    return x, k + 1


def sor(A, b, omega, tol=1e-8, max_iter=1000):
    x = np.zeros_like(b, dtype=float)

    for k in range(max_iter):
        x_og = x.copy()
        for i in range(A.shape[0]):
            sigma1 = np.dot(A[i, :i], x_og[:i])
            sigma2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_og[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sigma1 - sigma2)
        if np.linalg.norm(x - x_og, ord=np.inf) < tol:
            break
        x = x_og.copy()
    return x, k + 1


B = np.array([[4, 1, -1],
               [-1, 3, 1],
               [2, 2, 6]])
z = np.array([5, -4, 1])
print(jacobi(B, z))
print(gauss_seidel(B, z))
print(sor(B, z, 1.05))
print(sor(B, z, 0.95))


def matrix(n):
    A = ((n/2) * np.eye(n)) + np.random.normal(size=(n, n))
    return A


def sol(n):
    b = np.random.normal(size=(n, n))
    return b


def clock(n, type=0):
    A = matrix(n)
    b = sol(n)
    if type == 0:
        t0 = time.time()
        jacobi(A, b)
        t1 = time.time()
        total_time = t1 - t0
        return total_time, jacobi(A, b)[1]
    if type == 1:
        t0 = time.time()
        gauss_seidel(A, b)
        t1 = time.time()
        total_time = t1 - t0
        return total_time, gauss_seidel(A, b)[1]
    if type == 2:
        t0 = time.time()
        sor(A, b, 1.05)
        t1 = time.time()
        total_time = t1 - t0
        return total_time, sor(A, b, 1.05)[1]
    if type == 3:
        t0 = time.time()
        sor(A, b, 0.95)
        t1 = time.time()
        total_time = t1 - t0
        return total_time, sor(A, b, 0.95)[1]


def time_iter(type=0):
    times = np.empty((0, 0))
    iterations = np.empty((0, 0))
    if type == 0:
        times = np.append(times, (clock(10, 0)[0], clock(25, 0)[0], clock(50, 0)[0], clock(100, 0)[0],
                                  clock(200, 0)[0], clock(500, 0)[0]))
        iterations = np.append(iterations, (clock(10, 0)[1], clock(25, 0)[1], clock(50, 0)[1], clock(100, 0)[1],
                                            clock(200, 0)[1], clock(500, 0)[1]))
        return times, iterations
    if type == 1:
        times = np.append(times, (clock(10, 1)[0], clock(25, 1)[0], clock(50, 1)[0], clock(100, 1)[0],
                                  clock(200, 1)[0], clock(500, 1)[0]))
        iterations = np.append(iterations, (clock(10, 1)[1], clock(25, 1)[1], clock(50, 1)[1], clock(100, 1)[1],
                                            clock(200, 1)[1], clock(500, 1)[1]))
        return times, iterations
    if type == 2:
        times = np.append(times, (clock(10, 2)[0], clock(25, 2)[0], clock(50, 2)[0], clock(100, 2)[0],
                                  clock(200, 2)[0], clock(500, 2)[0]))
        iterations = np.append(iterations, (clock(10, 2)[1], clock(25, 2)[1], clock(50, 2)[1], clock(100, 2)[1],
                                            clock(200, 2)[1], clock(500, 2)[1]))
        return times, iterations
    if type == 3:
        times = np.append(times, (clock(10, 3)[0], clock(25, 3)[0], clock(50, 3)[0], clock(100, 3)[0],
                                  clock(200, 3)[0], clock(500, 3)[0]))
        iterations = np.append(iterations, (clock(10, 3)[1], clock(25, 3)[1], clock(50, 3)[1], clock(100, 3)[1],
                                            clock(200, 3)[1], clock(500, 3)[1]))
        return times, iterations


print(time_iter(type=0))
print(time_iter(type=1))
print(time_iter(type=2))
print(time_iter(type=3))

dims = [10, 25, 50, 100, 200, 500]
# size vs. time
plt.plot(dims, time_iter(type=0)[0])
plt.plot(dims, time_iter(type=1)[0])
plt.plot(dims, time_iter(type=2)[0])
plt.plot(dims, time_iter(type=3)[0])
plt.xlabel("n")
plt.ylabel("Wall Clock Time (seconds)")
plt.title("Execution Time of Solving Linear Systems")
plt.legend(["Jacobi Method", "Gauss-Seidel Method", "SOR Method (w=1.05)", "SOR Method (w=0.95)"], loc="upper left")
plt.show()

# size vs. iterations
plt.plot(dims, time_iter(type=0)[1])
plt.plot(dims, time_iter(type=1)[1])
plt.plot(dims, time_iter(type=2)[1])
plt.plot(dims, time_iter(type=3)[1])
plt.xlabel("n")
plt.ylabel("Number of Iterations")
plt.title("Number of Iterations for Solving Linear Systems")
plt.legend(["Jacobi Method", "Gauss-Seidel Method", "SOR Method (w=1.05)", "SOR Method (w=0.95)"], loc="upper left")
plt.show()


def residual(max_iter=1000, tol=1e-8, type=0):
    A = matrix(500)
    b = np.squeeze(sol(500))
    x = np.zeros_like(b, dtype=float)
    norms = np.empty((0, 0))
    iter = 0

    if type == 0:
        for k in range(max_iter):
            iter += 1
            r = b - A.dot(x)
            norm = np.linalg.norm(r, 2)
            norms = np.append(norms, norm)
            if norm < tol:
                break
            x[:] = (b - np.dot(A - np.diag(np.diagonal(A)), x)) / np.diagonal(A)
        return norms, iter
    if type == 1:
        for k in range(max_iter):
            iter += 1
            r = b - A.dot(x)
            norm = np.linalg.norm(r)
            norms = np.append(norms, norm)
            if norm < tol:
                break
            for i in range(A.shape[0]):
                x[i] = (b[i] - A[i, :i].dot(x[:i]) - A[i, i + 1:].dot(x[i + 1:])) / A[i, i]
        return norms, iter
    if type == 2:
        for k in range(max_iter):
            iter += 1
            r = b - A.dot(x)
            norm = np.linalg.norm(r)
            norms = np.append(norms, norm)
            if norm < tol:
                break
            for i in range(A.shape[0]):
                x[i] = (1 - 1.05) * x[i] + (1.05 / A[i, i]) * (
                            b[i] - A[i, :i].dot(x[:i]) - A[i, i + 1:].dot(x[i + 1:]))
        return norms, iter
    if type == 3:
        for k in range(max_iter):
            iter += 1
            r = b - A.dot(x)
            norm = np.linalg.norm(r)
            norms = np.append(norms, norm)
            if norm < tol:
                break
            for i in range(A.shape[0]):
                x[i] = (1 - 0.95) * x[i] + (0.95 / A[i, i]) * (
                        b[i] - A[i, :i].dot(x[:i]) - A[i, i + 1:].dot(x[i + 1:]))
        return norms, iter


# iteration vs. norm for 500x500 matrix
plt.plot(residual(type=0)[0], list(range(0, residual(type=0)[1])))
plt.plot(residual(type=1)[0], list(range(0, residual(type=1)[1])))
plt.plot(residual(type=2)[0], list(range(0, residual(type=2)[1])))
plt.plot(residual(type=3)[0], list(range(0, residual(type=3)[1])))
plt.xlabel("Iterations (k)")
plt.ylabel("Residual (2-norm)")
plt.title("Iteration vs. Residual")
plt.legend(["Jacobi Method", "Gauss-Seidel Method", "SOR Method (w=1.05)", "SOR Method (w=0.95)"], loc="upper right")
plt.show()
