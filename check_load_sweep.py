"""Cross-check of the COMSOL load sweep (Fig. 3m): same V-Q-x circuit, same FE Voc(x), C(x),
integrated exactly per step for 30 cycles so that even R*C >> period reaches steady state."""

import numpy as np

from teng_signals import load_comsol_table

xs, voc_t, c_t = load_comsol_table("data/comsol_fe_voc_c_vs_gap.csv")
S = 0.05 * 0.05
print("f_Hz,R_ohm,Vpeak_V,Ppeak_density_W_m2")
for f in (1.0, 8.0):
    T, n_cyc, steps = 1 / f, 30, 4000
    dt = T / steps
    t = np.arange(n_cyc * steps) * dt
    x = 1e-3 * (1 - np.cos(2 * np.pi * f * t)) / 2
    voc, c = np.interp(x, xs, voc_t), np.interp(x, xs, c_t)
    for R in (1e5, 1e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11):
        # exact for a linear ramp of the equilibrium charge Q_eq = C*Voc within each step,
        # so it stays correct when R*C is far below the time step
        q_eq = c * voc
        q = np.empty_like(t)
        q[0] = 0.0
        for i in range(1, len(t)):
            tau = R * c[i]
            s = (q_eq[i] - q_eq[i - 1]) / dt
            q[i] = q_eq[i] - tau * s + (q[i - 1] - q_eq[i - 1] + tau * s) * np.exp(-dt / tau)
        v = voc - q / c
        last = t >= (n_cyc - 1) * T
        print(f"{f},{R:.0e},{np.abs(v[last]).max():.3f},{(v[last] ** 2 / R).max() / S:.3e}")
