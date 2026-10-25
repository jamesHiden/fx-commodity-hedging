"""Illustrative Monte Carlo for a $100k coffee shipment paid in 3 months, rial revenue.
All parameters are assumptions for illustration, not calibrated estimates."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys, os

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ---------- assumptions ----------
N, DAYS, T = 20000, 63, 0.25
dt = T / DAYS
SIG_C = 0.35                       # arabica annual vol
MU_FX, SIG_FX = 0.25, 0.12         # rial depreciation trend, diffusion vol
LAM, J_MU, J_SD = 1.0, 0.15, 0.08  # jumps per year, log jump size
I_RIAL = 0.28                      # rial deposit rate (opportunity cost of buying USD early)
USD = 100                          # shipment, $k

# ---------- simulate ----------
zc = rng.standard_normal((N, DAYS))
coffee = np.exp(np.cumsum(-0.5 * SIG_C**2 * dt + SIG_C * np.sqrt(dt) * zc, axis=1))
zf = rng.standard_normal((N, DAYS))
jumps = rng.poisson(LAM * dt, (N, DAYS)) * rng.normal(J_MU, J_SD, (N, DAYS))
fx = np.exp(np.cumsum((MU_FX - 0.5 * SIG_FX**2) * dt + SIG_FX * np.sqrt(dt) * zf + jumps, axis=1))
coffee = np.hstack([np.ones((N, 1)), coffee])
fx = np.hstack([np.ones((N, 1)), fx])
cost_path = USD * coffee * fx      # rial cost index, today = 100
C, S = coffee[:, -1], fx[:, -1]

# ATM 3m call on coffee (Black-76), as fraction of price
d1 = 0.5 * SIG_C * np.sqrt(T)
PREM = norm.cdf(d1) - norm.cdf(-d1)
carry = 1 + I_RIAL * T

def cost(usd_needed, usd_bought_now):
    return usd_bought_now * carry + (usd_needed - usd_bought_now) * S

strategies = {
    "No hedge": cost(USD * C, 0),
    "Buy 50% USD now": cost(USD * C, 50),
    "Buy 100% USD now": cost(USD * C, 100),
    "100% USD now\n+ coffee call (cap)": cost(USD * np.minimum(C, 1) + USD * PREM, USD * (1 + PREM)),
    "Fix coffee price now\n+ 100% USD now": cost(np.full(N, USD), USD),
}

def es(x, q=0.95):
    v = np.quantile(x, q)
    return x[x >= v].mean()

# ---------- style ----------
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})
NOTE = "Illustrative parameters, not calibrated. Index: rial cost if bought today = 100."
days = np.arange(DAYS + 1)

# ---------- fig 1: Monte Carlo paths ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
panels = [(fx * 100, "USD/IRR rate (today = 100)"),
          (coffee * 100, "Coffee price (today = 100)"),
          (cost_path, "Rial cost of shipment (index)")]
for ax, (paths, title) in zip(axes, panels):
    for p in paths[:60]:
        ax.plot(days, p, color="#9a9994", lw=0.6, alpha=0.45)
    lo, med, hi = np.quantile(paths, [0.05, 0.5, 0.95], axis=0)
    ax.fill_between(days, lo, hi, color=BLUE, alpha=0.12, lw=0)
    ax.plot(days, med, color=BLUE, lw=2, label="Median")
    ax.plot(days, hi, color=BLUE, lw=1.2, ls="--", label="5th / 95th pct")
    ax.plot(days, lo, color=BLUE, lw=1.2, ls="--")
    ax.set_title(title)
    ax.set_xlabel("Trading days until payment")
    ax.annotate(f"95th: {hi[-1]:.0f}", (DAYS, hi[-1]), xytext=(-4, 6),
                textcoords="offset points", ha="right", color=INK, fontsize=9)
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Monte Carlo: 60 of 20,000 simulated paths over 3 months", x=0.01, ha="left",
             fontweight="bold", color=INK)
fig.text(0.01, -0.01, NOTE + " FX includes jumps (sudden devaluations).", color=INK2, fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT}/1_monte_carlo_paths.png", dpi=180, bbox_inches="tight")

# ---------- fig 2: distribution of final rial cost ----------
x = strategies["No hedge"]
mean, var95, es95 = x.mean(), np.quantile(x, 0.95), es(x)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.hist(x, bins=120, range=(50, 220), color=BLUE, alpha=0.85, edgecolor=SURF, linewidth=0.4)
top = ax.get_ylim()[1]
ax.axvline(100, color=INK, lw=1, alpha=0.5)
ax.text(98.5, top * 0.95, "Buy today: 100", color=INK2, fontsize=9, ha="right")
for v, lab, c, yf in [(mean, f"Expected: {mean:.0f}", INK2, 0.95), (var95, f"95% VaR: {var95:.0f}", ORANGE, 0.70),
                      (es95, f"Expected Shortfall: {es95:.0f}", ORANGE, 0.45)]:
    ax.axvline(v, color=c, lw=2, ls="-" if c == INK2 else "--")
    ax.text(v + 1.5, top * yf, lab, color=INK, fontsize=9,
            bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5))
ax.set_title("Distribution of rial cost at payment date, no hedge")
ax.set_xlabel("Rial cost index (buy today = 100)")
ax.set_ylabel("Number of simulations")
ax.grid(axis="x", visible=False)
fig.text(0.01, -0.02, NOTE + " Right tail = the bad outcomes.", color=INK2, fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT}/2_cost_distribution.png", dpi=180, bbox_inches="tight")

# ---------- fig 3: scenarios ----------
trend = np.exp(MU_FX * T)
fixed = USD * carry
hedged = lambda c, s: USD * (1 + PREM) * carry + USD * (min(c, 1) - 1) * s
scen = [
    ("Base: coffee flat, FX on trend", USD * 1.0 * trend, hedged(1.0, trend)),
    ("FX shock: USD +30%", USD * 1.0 * 1.30, hedged(1.0, 1.30)),
    ("Coffee shock: +40%", USD * 1.40 * trend, hedged(1.40, trend)),
    ("Both: coffee +40%, USD +30%", USD * 1.40 * 1.30, hedged(1.40, 1.30)),
    ("Coffee falls 25%, FX on trend", USD * 0.75 * trend, hedged(0.75, trend)),
]
fig, ax = plt.subplots(figsize=(9.5, 4.4))
y = np.arange(len(scen))[::-1]
h = 0.36
ax.barh(y + h / 2, [s[1] for s in scen], h, color=ORANGE, label="No hedge")
ax.barh(y - h / 2, [s[2] for s in scen], h, color=BLUE, label="100% USD now + coffee call")
for yi, s in zip(y, scen):
    ax.text(s[1] + 2, yi + h / 2, f"{s[1]:.0f}", va="center", fontsize=9, color=INK)
    ax.text(s[2] + 2, yi - h / 2, f"{s[2]:.0f}", va="center", fontsize=9, color=INK)
ax.set_yticks(y, [s[0] for s in scen])
ax.axvline(100, color=INK, lw=1)
ax.set_xlim(0, 200)
ax.set_xlabel("Rial cost index (buy today = 100)")
ax.set_title("Rial cost of the shipment under scenarios")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right")
fig.text(0.01, -0.02, NOTE + f" Call premium ~{PREM*100:.0f}% of price; rial deposit rate {I_RIAL*100:.0f}%/yr.",
         color=INK2, fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT}/3_scenarios.png", dpi=180, bbox_inches="tight")

# ---------- fig 4: strategy comparison ----------
fig, ax = plt.subplots(figsize=(9.5, 4.6))
names = list(strategies)
y = np.arange(len(names))[::-1]
for yi, n in zip(y, names):
    v = strategies[n]
    lo, med, hi = np.quantile(v, [0.05, 0.5, 0.95])
    e = es(v)
    ax.plot([lo, hi], [yi, yi], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(med, yi, "o", color=BLUE, ms=8, mec=SURF, mew=2)
    ax.plot(e, yi, "D", color=ORANGE, ms=7, mec=SURF, mew=1.5)
    ax.text(max(hi, e) + 3, yi, f"median {med:.0f} | ES {e:.0f}", va="center", fontsize=9, color=INK)
ax.plot([], [], "o", color=BLUE, label="Median, with 5th-95th pct range")
ax.plot([], [], "D", color=ORANGE, label="Expected Shortfall (worst 5% avg)")
ax.set_yticks(y, names)
ax.axvline(100, color=INK, lw=1, alpha=0.5)
ax.set_xlim(60, 230)
ax.set_xlabel("Rial cost index (buy today = 100)")
ax.set_title("Hedging strategies: expected cost vs. tail risk")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.text(0.01, -0.02, NOTE, color=INK2, fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT}/4_strategies.png", dpi=180, bbox_inches="tight")

print(f"call premium {PREM:.3f}, carry {carry:.3f}")
for n, v in strategies.items():
    print(n.replace("\n", " "), np.round(np.quantile(v, [0.05, 0.5, 0.95]), 1), "mean", round(v.mean(), 1), "ES", round(es(v), 1))
for s in scen:
    print(s[0], round(s[1], 1), round(s[2], 1))
