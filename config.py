"""
config.py — every parameter of the study (v3, stress-dependent reliability).
Nothing downstream hard-codes a number.
"""
SYSTEMS = ["IEEE39", "IEEE57", "IEEE118"]
DT = 1.0                      # h
WARMUP_YEARS = 1              # simulated before each study year (stationary start)

# ------------------------------------------------------------ system construction
MAX_UNIT_MW = 400.0           # PGLib generators split into identical units <= 400 MW
LOLE_TARGET_H = 9.39418       # peak set so conventional HL-I LOLE = IEEE RTS-79 value (h/yr)

# ------------------------------------------------------------ stress model
COUPLING_EXP = 1.0            # kappa: stress ~ (L(t)/L_peak)^kappa  (1 = linear loading factor)
STRESS_RANGES = {             # per-component factors, drawn once (seeded, exported)
    "unit":        {"eta_L": (0.9, 1.3), "T": (40, 70), "eta_A": (0.8, 1.6), "eta_V": (0.8, 1.2)},
    "transformer": {"eta_L": (0.8, 1.2), "T": (45, 80), "eta_A": (0.9, 1.7), "eta_V": (0.9, 1.3)},
    "line":        {"eta_L": (0.7, 1.4), "T": (20, 45), "eta_A": (0.6, 1.4), "eta_V": (0.9, 1.2)},
}
T_REF = 60.0                  # degC
TTF_EPS = 0.05                # epsilon inside the denominator of TTF
ALPHA_FRACTION = 0.2          # alpha = 20 % of MTTF  (hazard <= 5 / MTTF)
GAMMA_FRACTION = 0.5          # gamma: stress-independent share of the mean repair time

LOL_TOL_MW = 1e-3
UNLIMITED_RATING_MW = 9000.0

# ------------------------------------------------------------ experiments
YEARS_MAIN = 1000
BASE_SEED = 20260930

# ------------------------------------------------------------ storage planning
VOLL = 10_000.0               # $/MWh value of lost load
STORAGE_CAPEX = 1.2e6         # $/MW for a 4-h battery (incl. energy), overnight
CRF = 0.1424                  # capital recovery factor, 7 %, 10 yr
STORAGE_HOURS = 4.0
ETA_CH, ETA_DIS = 0.95, 0.95
LOLE_TARGET_PLAN = 2.4        # h/yr reliability target (0.1 day/yr)
LOLE_PENALTY = 2.0e6          # $ per hour of LOLE above target (soft constraint)
