import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fractions import Fraction

def f_q1(x, b=0.001):
    x1, x2, x3 = x
    return 0.25 * x1**4 + (b / 3.0) * x2**3 + b * np.log(1.0 + x3**2)

def grad_q1(x, b=0.001):
    x1, x2, x3 = x
    g1 = x1**3
    g2 = b * x2**2
    g3 = (2.0 * b * x3) / (1.0 + x3**2)
    return np.array([g1, g2, g3])

def run_gradient_descent(x0, eta, b, iterations):
    history = [np.array(x0, dtype=float)]
    x = np.array(x0, dtype=float)
    for _ in range(iterations):
        g = grad_q1(x, b)
        x = x - eta * g
        history.append(x.copy())
    return np.array(history)

def verify_q2_newton(b):
    # Rosenbrock f(x) = (1 - x1)^2 + b*(x2 - x1^2)^2
    # grad f = [-2*(1 - x1) - 4*b*x1*(x2 - x1^2), 2*b*(x2 - x1^2)]
    # Hessian:
    # H11 = 2 - 4*b*x2 + 12*b*x1^2
    # H12 = -4*b*x1
    # H22 = 2*b
    x = np.array([-1.2, 1.0])
    print(f"--- Q2 Newton's Method Verification (b = {b}) ---")
    print(f"Starting point x^(0) = {x}")
    for i in range(2 if b==1 else 1):
        x1, x2 = x
        g1 = -2.0 * (1.0 - x1) - 4.0 * b * x1 * (x2 - x1**2)
        g2 = 2.0 * b * (x2 - x1**2)
        g = np.array([g1, g2])
        
        H11 = 2.0 - 4.0 * b * x2 + 12.0 * b * x1**2
        H12 = -4.0 * b * x1
        H22 = 2.0 * b
        H = np.array([[H11, H12], [H12, H22]])
        
        det_H = H11*H22 - H12**2
        cond = np.linalg.cond(H)
        
        # Newton step: - H^-1 * g
        step = -np.linalg.solve(H, g)
        x_new = x + step
        print(f"Iteration {i+1}:")
        print(f"  Gradient g^({i}) = {g}")
        print(f"  Hessian H^({i}) = \n{H}")
        print(f"  Det(H) = {det_H:.4f}, Condition Number = {cond:.4f}")
        print(f"  Newton Step s^({i}) = {step}")
        print(f"  Updated x^({i+1}) = {x_new}")
        x = x_new
    print()

def q2_quasi_newton_exact():
    print("--- Q2(d) & (e) Quasi-Newton Analytical Verification ---")
    S = np.array([[4, 1], [1, 2]], dtype=float)
    x0 = np.array([1, 1], dtype=float)
    H0 = np.eye(2)
    
    # Gradient g0 = S * x0
    g0 = S @ x0
    print(f"g0 = {g0}")
    
    # d0 = - H0 * g0 = - g0
    d0 = - H0 @ g0
    print(f"d0 = {d0}")
    
    # Exact line search for quadratic f(x) = 0.5 * x^T S x:
    # alpha0 = - (g0^T d0) / (d0^T S d0)
    alpha0 = - (g0.T @ d0) / (d0.T @ S @ d0)
    print(f"alpha0 = {alpha0} (as Fraction: {Fraction(alpha0).limit_denominator()})")
    
    x1 = x0 + alpha0 * d0
    print(f"x1 = {x1}")
    
    g1 = S @ x1
    print(f"g1 = {g1}")
    
    s0 = x1 - x0
    y0 = g1 - g0
    print(f"s0 = {s0}")
    print(f"y0 = {y0}")
    
    # BFGS update for inverse Hessian H1
    # H1 = (I - rho * s0 * y0^T) H0 (I - rho * y0 * s0^T) + rho * s0 * s0^T
    rho = 1.0 / np.dot(y0, s0)
    print(f"rho = {rho} (as Fraction: {Fraction(rho).limit_denominator()})")
    
    I = np.eye(2)
    term1 = I - rho * np.outer(s0, y0)
    term2 = I - rho * np.outer(y0, s0)
    H1_bfgs = term1 @ H0 @ term2 + rho * np.outer(s0, s0)
    print("H1 BFGS:")
    print(H1_bfgs)
    for r in range(2):
        for c in range(2):
            val = H1_bfgs[r, c]
            frac = Fraction(val).limit_denominator(100000)
            print(f"  H1_bfgs[{r},{c}] = {frac}")
            
    # SR1 update
    # H1 = H0 + ((s0 - H0 y0)(s0 - H0 y0)^T) / ((s0 - H0 y0)^T y0)
    diff = s0 - H0 @ y0
    denom = np.dot(diff, y0)
    H1_sr1 = H0 + np.outer(diff, diff) / denom
    print("H1 SR1:")
    print(H1_sr1)
    for r in range(2):
        for c in range(2):
            val = H1_sr1[r, c]
            frac = Fraction(val).limit_denominator(100000)
            print(f"  H1_sr1[{r},{c}] = {frac}")
    print()

def generate_q1_plot():
    x0 = [2.0, 2.0, 2.0]
    b = 0.001
    eta = 0.2
    iterations = 50
    history = run_gradient_descent(x0, eta, b, iterations)
    
    plt.figure(figsize=(10, 6), dpi=150)
    iterations_range = np.arange(iterations + 1)
    
    plt.plot(iterations_range, history[:, 0], label=r'$x_1$', color='#1f77b4', linewidth=2.5)
    plt.plot(iterations_range, history[:, 1], label=r'$x_2$', color='#ff7f0e', linewidth=2)
    plt.plot(iterations_range, history[:, 2], label=r'$x_3$', color='#2ca02c', linewidth=2)
    
    plt.title(r'Gradient Descent Convergence on $f(x)$ for $\eta = 0.2, b = 0.001$', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Variable Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    
    plot_path = '/workspace/scratch/gradient_descent_q1e.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved successfully to {plot_path}")

if __name__ == "__main__":
    generate_q1_plot()
    print()
    verify_q2_newton(b=1)
    verify_q2_newton(b=1000)
    q2_quasi_newton_exact()
