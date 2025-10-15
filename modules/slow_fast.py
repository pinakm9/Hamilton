import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import utility as ut

# --------------------------
# 1) Systems (direct port of MATLAB)
# --------------------------

# --------------------------
# 2) Parameters & balanced initial conditions
# --------------------------
J = np.array([[0.0, 1.0], [-1.0, 0.0]])
PARAMS = [6.0, 0.0, 0.1, -4.5, 0.0, -4.5, 0.001]  # [A11, A12, A22, B11, B12, B22, eps]

def gradV_point(q, params):
    """Return ∇V(q) for a single point q of shape (2,)."""
    A11, A12, A22, B11, B12, B22 = params[:6]
    q1, q2 = float(q[0]), float(q[1])
    g1 = A11*q1**3 + A12*q1*q2**2 + B11*q1 + B12*q2
    g2 = A22*q2**3 + A12*q2*q1**2 + B22*q2 + B12*q1
    return np.array([g1, g2], dtype=float)

# Balanced initial conditions
q0 = np.array([0.0, -9.8], dtype=float)
p0 = -J @ gradV_point(q0, PARAMS)       # <- note: slice NOT needed here because gradV_point slices internally
x0 = np.hstack([q0, p0])                

def _gradV_point(q1, q2, A11, A12, A22, B11, B12, B22):
    g1 = A11*q1**3 + A12*q1*q2**2 + B11*q1 + B12*q2
    g2 = A22*q2**3 + A12*q2*q1**2 + B22*q2 + B12*q1
    return np.array([g1, g2])

def redHamSys(t, q, params=PARAMS, J=J):
    # Supports q shape (2,) or (2, M)
    A11, A12, A22, B11, B12, B22 = params[:6]
    q = np.asarray(q)

    if q.ndim == 1:
        q1, q2 = q
        gradv = _gradV_point(q1, q2, A11, A12, A22, B11, B12, B22)
        return -J @ gradv

    # vectorized (2, M): operate columnwise
    q1, q2 = q[0, :], q[1, :]
    g1 = A11*q1**3 + A12*q1*q2**2 + B11*q1 + B12*q2
    g2 = A22*q2**3 + A12*q2*q1**2 + B22*q2 + B12*q1
    gradv = np.vstack((g1, g2))
    return -J @ gradv

def fullHamSys(t, x, params=PARAMS):
    # x shape (4,) or (4, M); component form is simplest and safe
    A11, A12, A22, B11, B12, B22, eps = params
    x = np.asarray(x)

    if x.ndim == 1:
        q1, q2, p1, p2 = x
        return np.array([
            p1,
            p2,
            (-A11*q1**3 - A12*q1*q2**2 - B11*q1 - B12*q2 + p2) / eps,
            (-A22*q2**3 - A12*q2*q1**2 - B22*q2 - B12*q1 - p1) / eps
        ])

    # vectorized (4, M)
    q1, q2, p1, p2 = x[0, :], x[1, :], x[2, :], x[3, :]
    qdot0 = p1
    qdot1 = p2
    qdot2 = (-A11*q1**3 - A12*q1*q2**2 - B11*q1 - B12*q2 + p2) / eps
    qdot3 = (-A22*q2**3 - A12*q2*q1**2 - B22*q2 - B12*q1 - p1) / eps
    return np.vstack((qdot0, qdot1, qdot2, qdot3))

def gradV(q, params=PARAMS):
    """
    Compute ∇V(q) for q of shape (N, 2),
    where each row is [q1, q2].

    Parameters
    ----------
    q : np.ndarray
        Shape (N, 2), each row = [q1, q2].
    params : sequence
        [A11, A12, A22, B11, B12, B22].

    Returns
    -------
    gradV : np.ndarray
        Shape (N, 2), each row = ∇V(q_i).
    """
    A11, A12, A22, B11, B12, B22 = params[:6]
    q1, q2 = q[:, 0], q[:, 1]

    grad_q1 = A11*q1**3 + A12*q1*q2**2 + B11*q1 + B12*q2
    grad_q2 = A22*q2**3 + A12*q2*q1**2 + B22*q2 + B12*q1

    return np.column_stack((grad_q1, grad_q2))



def gradV_Nx2(q, params=PARAMS):
    A11, A12, A22, B11, B12, B22 = params[:6]
    q1, q2 = q[:, 0], q[:, 1]
    g1 = A11*q1**3 + A12*q1*q2**2 + B11*q1 + B12*q2
    g2 = A22*q2**3 + A12*q2*q1**2 + B22*q2 + B12*q1
    return np.column_stack((g1, g2))



def balance(q, p, params=PARAMS, J=J):
    g = gradV_Nx2(q, params)          # (N,2)
    term = p + g @ J.T                # (N,2)  (row-oriented => J.T)
    # term = p @ J.T - g   
    # bal = np.einsum('ij,ij->i', term, term)
    return np.sum(term**2, axis=1)


Tend = 20.0
dts  = 0.01
t_eval = np.arange(0.0, Tend + dts, dts)

# --------------------------
# 3) Integrations
# --------------------------

@ut.timer
def solve_full(x0=x0, Tend=Tend, N=20000):
    dynamical_system = lambda t, x: fullHamSys(t, x, PARAMS)
    t_eval = np.linspace(0.0, Tend, N)
    return solve_ivp(dynamical_system, (0.0, Tend), x0, t_eval=t_eval, rtol=1e-8, atol=1e-8, method='Radau')


@ut.timer
def solve_red(q0=q0, Tend=Tend, N=20000):
    dynamical_system = lambda t, x: redHamSys(t, x, PARAMS)
    t_eval = np.linspace(0.0, Tend, N)
    return solve_ivp(dynamical_system, (0.0, Tend), q0, t_eval=t_eval, rtol=1e-8, atol=1e-8, method='RK45')




