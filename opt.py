
from collections import namedtuple
import numpy as np


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""


def gauss_newton(y, f, j, x0, k=1, tol=1e-4):
    x = x0.copy()
    n_iter = 0
    converged = False

    while not converged:
        n_iter += 1
        fx = f(*x)
        Jx = j(*x)
        r = y - fx
        delta_x = np.linalg.lstsq(Jx, -k*r, rcond=None)[0]
        x += delta_x

        if np.linalg.norm(delta_x)/np.linalg.norm(x) < tol:
            converged = True

        if n_iter > 100:
            break

    return Result(x, n_iter, converged)    
    pass

def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
    x = x0.copy()
    lmbd = lmbd0
    n_iter = 0
    converged = False

    while not converged:
        n_iter += 1
        fx = f(*x)
        Jx = j(*x)
        r = y - fx
        JTr = Jx.T @ r
        JJT = Jx.T @ Jx
        delta_x = np.linalg.solve(JJT + lmbd*np.diag(np.diag(JJT)), -JTr)
        x_new = x + delta_x
        fx_new = f(*x_new)
        r_new = y - fx_new

        if np.linalg.norm(r_new) < np.linalg.norm(r):
            lmbd /= nu
            x = x_new
            r = r_new
        else:
            lmbd *= nu

        if np.linalg.norm(delta_x)/np.linalg.norm(x) < tol:
            converged = True

        if n_iter > 100:
            break

    return Result(x, n_iter, converged)
    pass

if __name__ == "__main__":
    pass
