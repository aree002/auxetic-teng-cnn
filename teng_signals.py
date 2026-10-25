"""Synthetic TENG grasp signals built on the COMSOL contact-separation model.

The device physics comes from COMSOL (Sim 6c): open-circuit voltage Voc(x) and
capacitance C(x) of the 5 x 5 cm FEP / CA-PEI TENG versus gap x. A grasp moves the
gap x(t); the voltage across the measurement load R follows the V-Q-x circuit

    R dQ/dt = Voc(x) - Q / C(x),      V = R dQ/dt.

Six grasp classes (cf. Fig. 6j: human hand and robotic gripper, three objects each)
differ in effective tribo-charge (contact pressure / area), press speed, hold time,
rhythm and timing jitter. The auxetic and non-auxetic devices differ in how much
contact area survives a curved grasp and in signal noise (Figs. 6b-g).
"""

import numpy as np

FS = 500            # sampling rate of the stored signals (Hz)
WINDOW_S = 2.048    # window length (s) -> 1024 samples
DT_SIM = 0.25e-3    # circuit integration step (s)
R_LOAD = 1e8        # measurement load (ohm)

CLASS_NAMES = [
    "1 hand: light touch",
    "2 hand: fist",
    "3 hand: orange",
    "4 gripper: empty",
    "5 gripper: orange",
    "6 gripper: apple",
]

# s = fraction of the calibrated tribo-charge reached (pressure / contact area)
# press/hold in s, period = mean time between grasps, jitter = relative timing spread
CLASSES = [
    dict(s=0.12, x0=1.0e-3, press=0.060, hold=0.15, period=0.60, jitter=0.25),
    dict(s=0.22, x0=0.8e-3, press=0.045, hold=0.25, period=0.70, jitter=0.25),
    dict(s=0.34, x0=1.0e-3, press=0.050, hold=0.30, period=0.80, jitter=0.25),
    dict(s=0.12, x0=1.0e-3, press=0.030, hold=0.20, period=0.50, jitter=0.04),
    dict(s=0.22, x0=0.8e-3, press=0.030, hold=0.30, period=0.60, jitter=0.04),
    dict(s=0.34, x0=1.0e-3, press=0.030, hold=0.30, period=0.60, jitter=0.04),
]

DEVICES = {
    # auxetic: conformal contact keeps the charge; low noise (Fig. 6e, 6g)
    "auxetic": dict(contact=1.00, event_spread=0.08, noise_v=0.8, hum_v=0.4),
    # non-auxetic: edge delamination loses contact area unevenly; noisy (Fig. 6d, 6f)
    "non_auxetic": dict(contact=0.60, event_spread=0.30, noise_v=2.0, hum_v=1.0),
}


def load_comsol_table(path):
    """FE table from COMSOL: gap_mm, Voc_V, Qsc_nC, C_pF."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    x = np.concatenate([[0.0], d["gap_mm"] * 1e-3])
    voc = np.concatenate([[0.0], d["Voc_V"]])
    c = d["C_pF"] * 1e-12
    c0 = c[0] + (c[0] - c[1]) * (d["gap_mm"][0] / (d["gap_mm"][1] - d["gap_mm"][0]))
    c = np.concatenate([[c0], c])
    return x, voc, c


def gap_profile(t, rng, p):
    """Gap x(t): separated at x0, smooth press to contact, hold, smooth release."""
    x = np.full_like(t, p["x0"])
    tc = rng.uniform(0.05, 0.35)
    while tc < t[-1]:
        press = p["press"] * rng.uniform(0.8, 1.2)
        hold = p["hold"] * rng.uniform(0.8, 1.2)
        release = press * rng.uniform(1.0, 1.6)
        t1, t2, t3 = tc + press, tc + press + hold, tc + press + hold + release
        m = (t >= tc) & (t < t1)
        x[m] = p["x0"] * 0.5 * (1 + np.cos(np.pi * (t[m] - tc) / press))
        x[(t >= t1) & (t < t2)] = 0.0
        m = (t >= t2) & (t < t3)
        x[m] = p["x0"] * 0.5 * (1 - np.cos(np.pi * (t[m] - t2) / release))
        tc += p["period"] * (1 + p["jitter"] * rng.standard_normal())
        tc = max(tc, t3 + 0.02)
    return x


def simulate_voltage(x, frac, table):
    """Integrate R dQ/dt = Voc(x) - Q/C(x) exactly per step (piecewise-constant Voc, C)."""
    xs, voc_t, c_t = table
    voc = frac * np.interp(x, xs, voc_t)
    c = np.interp(x, xs, c_t)
    q = np.empty_like(x)
    q[0] = c[0] * voc[0]
    for i in range(1, len(x)):
        q_eq = c[i] * voc[i]
        q[i] = q_eq + (q[i - 1] - q_eq) * np.exp(-DT_SIM / (R_LOAD * c[i]))
    return voc - q / c


def make_signal(cls, device, rng, table):
    p = CLASSES[cls]
    dv = DEVICES[device]
    t = np.arange(0, WINDOW_S, DT_SIM)
    x = gap_profile(t, rng, p)
    frac = p["s"] * dv["contact"] * rng.uniform(0.85, 1.15)
    v = simulate_voltage(x, frac, table)
    # per-event amplitude spread (contact loss differs grasp to grasp)
    events = np.cumsum(np.r_[0, np.diff((x == 0).astype(int)) == 1])
    gain = 1 + dv["event_spread"] * rng.standard_normal(events.max() + 1)
    v = v * gain[events]
    step = int(round(1 / (FS * DT_SIM)))
    v = v[::step][: int(WINDOW_S * FS)]
    ts = np.arange(len(v)) / FS
    v += dv["noise_v"] * rng.standard_normal(len(v))
    v += dv["hum_v"] * np.sin(2 * np.pi * 50 * ts + rng.uniform(0, 2 * np.pi))
    v += rng.normal(0, 0.3) + rng.normal(0, 0.5) * ts / WINDOW_S   # offset + drift
    return v.astype(np.float32)


def make_dataset(table, device="auxetic", n_per_class=300, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls in range(len(CLASSES)):
        for _ in range(n_per_class):
            X.append(make_signal(cls, device, rng, table))
            y.append(cls)
    return np.stack(X), np.array(y)
