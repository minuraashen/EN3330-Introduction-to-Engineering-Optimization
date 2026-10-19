"""
EN3330 Assignment 03 - plotting script
Generates every figure used in the report and saves each one as both
PDF (vector, used by LaTeX) and PNG into the ./figures folder.

Run:  python plots.py
Requires: numpy, matplotlib
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
cols = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    """Save a figure as PDF (for LaTeX) and PNG (for quick viewing)."""
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)


# Lagrangian and dual function of Question 1

L = lambda mu, x: x**2 + mu * (1 - x)
d = lambda mu: mu - mu**2 / 4

# ---------- Q1(a) ----------
x = np.linspace(-1.5, 3, 400)
fig, ax = plt.subplots(figsize=(7, 4.5))
for mu, c in zip([0, 1, 2, 3], cols):
    ax.plot(x, L(mu, x), color=c, label=f"μ = {mu}")
    xm = mu / 2
    ax.plot(xm, L(mu, xm), "o", color=c, ms=8, mec="k")
    ax.annotate(f"({xm:.1f}, {L(mu, xm):.2f})", (xm, L(mu, xm)),
                textcoords="offset points", xytext=(6, -16), fontsize=9, color=c)
ax.axvline(1, color="gray", ls="--", lw=1)
ax.text(1.03, 7.5, "constraint boundary x = 1", color="gray", fontsize=9)
ax.axvspan(-1.5, 1, color="red", alpha=0.05)
ax.text(-1.4, 7.5, "infeasible (x < 1)", color="red", fontsize=9)
ax.set_xlim(-1.5, 3); ax.set_ylim(-1, 8.5)
ax.set_xlabel("x"); ax.set_ylabel("L(μ, x)")
ax.set_title("Lagrangian L(μ, x) = x² + μ(1 − x) for μ = 0, 1, 2, 3")
ax.legend(loc="lower right", fontsize=9)
save(fig, "q1a")

# ---------- Q1(b) ----------
mu = np.linspace(0, 5, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(mu, d(mu), color="tab:blue", label="d(μ) = μ − μ²/4")
ax.axhline(1, color="tab:red", ls="--", lw=1, label="primal optimum p* = 1")
ax.plot(2, 1, "o", color="tab:red", ms=9, mec="k")
ax.annotate("maximum: μ* = 2, d* = 1", (2, 1), xytext=(2.4, 0.35),
            arrowprops=dict(arrowstyle="->"))
for m in [0, 1, 3]:
    ax.plot(m, d(m), "s", color="tab:blue", ms=6)
ax.set_xlabel("μ"); ax.set_ylabel("d(μ)")
ax.set_ylim(-1.5, 1.5)
ax.set_title("Dual function d(μ)")
ax.legend(loc="lower left")
save(fig, "q1b")

# ---------- Q1(c) ----------
x = np.linspace(-1, 3, 400)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axs[0]
mus = [0, 1, 2, 5, 10, 20]
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(mus)))
for m, c in zip(mus, cmap):
    ax.plot(x, L(m, x), color=c, label=f"μ = {m}")
ax.axvline(1, color="gray", ls="--", lw=1)
ax.set_ylim(-2, 25)
ax.set_xlabel("x"); ax.set_ylabel("L(μ, x)")
ax.set_title("L(μ, x) for growing μ")
ax.text(-0.9, 22, "x < 1: curves keep rising\nas μ grows → max = +∞", fontsize=9)
ax.text(1.4, 15, "x ≥ 1: largest value\nis at μ = 0 → x²", fontsize=9)
ax.legend(fontsize=8, loc="upper right", ncol=2)

ax = axs[1]
xf = x[x >= 1]
ax.axvspan(-1, 1, color="red", alpha=0.15)
ax.text(-0.75, 6.5, "g(x) = +∞\n(infeasible)", color="darkred", fontsize=11)
ax.annotate("", xy=(0, 9.6), xytext=(0, 8.5),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=2))
ax.plot(xf, xf**2, color="tab:blue", lw=2.5, label="g(x) = x² (feasible)")
ax.plot(x, x**2, color="tab:blue", ls=":", lw=1, label="x² (for reference)")
ax.plot(1, 1, "o", color="tab:red", ms=9, mec="k")
ax.annotate("min g = 1 at x = 1", (1, 1), xytext=(1.5, 0.3),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlim(-1, 3); ax.set_ylim(0, 10)
ax.set_xlabel("x"); ax.set_ylabel("g(x)")
ax.set_title("Primal function g(x) = max over μ ≥ 0 of L(μ, x)")
ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0.52, 1.0))
fig.tight_layout()
save(fig, "q1c")

# ---------- Q2(a)(iv) ----------
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
mu = np.linspace(0, 5, 100)
axs[0].plot(mu, np.ones_like(mu), color="tab:blue", lw=2, label="L(μ, x*) = 1")
axs[0].plot(2, 1, "o", color="tab:red", ms=9, mec="k", label="L(μ*, x*) = 1")
axs[0].set_ylim(0, 2); axs[0].set_xlabel("μ"); axs[0].set_ylabel("L(μ, 1)")
axs[0].set_title("L(μ, x*) vs μ  (x* = 1)")
axs[0].legend()
x = np.linspace(-1, 3, 300)
axs[1].plot(x, (x - 1)**2 + 1, color="tab:green", lw=2, label="L(μ*, x) = (x−1)² + 1")
axs[1].axhline(1, color="gray", ls="--", lw=1)
axs[1].plot(1, 1, "o", color="tab:red", ms=9, mec="k", label="L(μ*, x*) = 1")
axs[1].set_xlabel("x"); axs[1].set_ylabel("L(2, x)")
axs[1].set_title("L(μ*, x) vs x  (μ* = 2)")
axs[1].legend()
fig.tight_layout()
save(fig, "q2a")

# ---------- Q2(c) new constraint x >= -1 ----------
fig, ax = plt.subplots(figsize=(6.5, 3.6))
x = np.linspace(-2, 2, 300)
ax.plot(x, x**2, color="tab:blue", lw=2, label="f(x) = x²")
ax.axvspan(-2, -1, color="red", alpha=0.12, label="infeasible (x < −1)")
ax.axvline(-1, color="gray", ls="--", lw=1)
ax.plot(0, 0, "o", color="tab:red", ms=9, mec="k", label="x* = 0 (inside, not on boundary)")
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.set_title("Modified problem: min x²  s.t.  x ≥ −1")
ax.legend(fontsize=9)
save(fig, "q2c")

# ---------- Q3(a) ----------
fig, ax = plt.subplots(figsize=(6.5, 3.6))
mus = [0, 1, 2, 3]
vals = [d(m) for m in mus]
bars = ax.bar([str(m) for m in mus], vals, color="tab:blue", alpha=0.8, label="d(μ)")
ax.axhline(1, color="tab:red", ls="--", label="p* = 1")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.2f}", ha="center")
ax.set_xlabel("μ"); ax.set_ylabel("value"); ax.set_ylim(0, 1.3)
ax.set_title("Weak duality check: d(μ) ≤ p*")
ax.legend(loc="upper left")
save(fig, "q3a")

# ---------- Q3(b) ----------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
mu = np.linspace(0, 5, 400)
for xv, c in zip([0, 1, 2], cols):
    ax.plot(mu, L(mu, xv), color=c, lw=1.5, label=f"x = {xv}:  L = {['μ', '1', '4 − μ'][xv]}")
env = np.minimum.reduce([L(mu, 0), L(mu, 1), L(mu, 2)])
ax.plot(mu, env, color="k", lw=3, alpha=0.6, label="lower envelope (min of 3 lines)")
ax.plot(mu, d(mu), color="tab:purple", ls="--", lw=2, label="true d(μ) = μ − μ²/4")
ax.set_ylim(-1.5, 4.5)
ax.set_xlabel("μ"); ax.set_ylabel("L(μ, x)")
ax.set_title("L(μ, x) as a function of μ for fixed x")
ax.legend(fontsize=9, loc="upper right")
save(fig, "q3b")
print("All figures saved to ./" + OUT)