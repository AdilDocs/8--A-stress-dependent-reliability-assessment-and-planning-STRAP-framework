"""
data_sources.py — benchmark data used by the study, all from published sources.

1. Test networks: IEEE PES PGLib-OPF v23.07 (Babaeinejadsarookolaee et al.,
   arXiv:1908.02788) — pglib_opf_case39_epri, case57_ieee, case118_ieee.
   These carry thermal ratings (rateA) for every branch, unlike the MATPOWER /
   pandapower originals of the 57- and 118-bus cases.
2. Chronological load: IEEE Reliability Test System 1979 (IEEE RTS Task Force,
   IEEE Trans. PAS-98, 1979), Tables 1-3: weekly, daily and hourly factors,
   52 x 7 x 24 = 8736 hours.
3. Reliability data: IEEE RTS-79 Table 6 (generating units by size) and
   Tables 9-10 (overhead lines, transformers).
"""
import os
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# IEEE RTS-79 chronological load model
# ---------------------------------------------------------------------------
WEEKLY_PEAK = np.array([
    86.2, 90.0, 87.8, 83.4, 88.0, 84.1, 83.2, 80.6, 74.0, 73.7, 71.5, 72.7, 70.4,
    75.0, 72.1, 80.0, 75.4, 83.7, 87.0, 88.0, 85.6, 81.1, 90.0, 88.7, 89.6, 86.1,
    75.5, 81.6, 80.1, 88.0, 72.2, 77.6, 80.0, 72.9, 72.6, 70.5, 78.0, 69.5, 72.4,
    72.4, 74.3, 74.4, 80.0, 88.1, 88.5, 90.9, 94.0, 89.0, 94.2, 97.0, 100.0, 95.2]) / 100
DAILY_PEAK = np.array([93, 100, 98, 96, 94, 77, 75]) / 100          # Mon..Sun
HOURLY = {  # % of daily peak, hour 1..24
    ("winter", "wd"): [67, 63, 60, 59, 59, 60, 74, 86, 95, 96, 96, 95, 95, 95, 93, 94, 99, 100, 100, 96, 91, 83, 73, 63],
    ("winter", "we"): [78, 72, 68, 66, 64, 65, 66, 70, 80, 88, 90, 91, 90, 88, 87, 87, 91, 100, 99, 97, 94, 92, 87, 81],
    ("summer", "wd"): [64, 60, 58, 56, 56, 58, 64, 76, 87, 95, 99, 100, 99, 100, 100, 97, 96, 96, 93, 92, 92, 93, 87, 72],
    ("summer", "we"): [74, 70, 66, 65, 64, 62, 62, 66, 81, 86, 91, 93, 93, 92, 91, 91, 92, 94, 95, 95, 100, 93, 88, 80],
    ("spring", "wd"): [63, 62, 60, 58, 59, 65, 72, 85, 95, 99, 100, 99, 93, 92, 90, 88, 90, 92, 96, 98, 96, 90, 80, 70],
    ("spring", "we"): [75, 73, 69, 66, 65, 65, 68, 74, 83, 89, 92, 94, 91, 90, 90, 86, 85, 88, 92, 100, 97, 95, 90, 85],
}


def _season(week):  # week 1..52
    if week <= 8 or week >= 44:
        return "winter"
    if 18 <= week <= 30:
        return "summer"
    return "spring"


def rts_load_profile():
    """Hourly load in per unit of annual peak, 8736 values (RTS-79)."""
    out = np.empty(52 * 7 * 24)
    k = 0
    for w in range(1, 53):
        s = _season(w)
        for d in range(7):
            day = "we" if d >= 5 else "wd"
            hp = np.array(HOURLY[(s, day)]) / 100
            out[k:k + 24] = WEEKLY_PEAK[w - 1] * DAILY_PEAK[d] * hp
            k += 24
    return out


# ---------------------------------------------------------------------------
# IEEE RTS-79 reliability data
# ---------------------------------------------------------------------------
# unit size (MW): (MTTF h, MTTR h)
RTS_UNITS = {12: (2940, 60), 20: (450, 50), 50: (1980, 20), 76: (1960, 40),
             100: (1200, 50), 155: (960, 40), 197: (950, 50), 350: (1150, 100),
             400: (1100, 150)}
RTS79_SYSTEM = [12] * 5 + [20] * 4 + [50] * 6 + [76] * 4 + [100] * 3 + [155] * 4 + \
               [197] * 3 + [350] + [400] * 2          # 32 units, 3405 MW
RTS79_PEAK = 2850.0
# Published HL-I indices of RTS-79 with the hourly load model
RTS79_REF = dict(LOLE_h=9.39418, EENS_MWh=1176.0)

# Branches (RTS-79 Tables 9-10, class averages): failure rate /yr, MTTR h
LINE_DATA = dict(lam_per_yr=0.5, mttr_h=10.0)
TRAFO_DATA = dict(lam_per_yr=0.02, mttr_h=768.0)


def unit_reliability(pmax):
    """(MTTF, MTTR) of the RTS unit class nearest in size (log scale)."""
    sizes = np.array(sorted(RTS_UNITS))
    k = sizes[np.argmin(np.abs(np.log(sizes) - np.log(max(pmax, 1.0))))]
    return RTS_UNITS[int(k)]


# ---------------------------------------------------------------------------
# PGLib-OPF MATPOWER parser
# ---------------------------------------------------------------------------
def _matrix(text, name):
    m = re.search(r"mpc\.%s\s*=\s*\[(.*?)\];" % name, text, re.S)
    rows = []
    for line in m.group(1).split("\n"):
        line = line.split("%")[0].strip().rstrip(";").strip()
        if line:
            rows.append([float(v) for v in line.replace(";", " ").split()])
    return np.array(rows)


PGLIB_FILES = {"IEEE39": "pglib_opf_case39_epri.m", "IEEE57": "pglib_opf_case57_ieee.m",
               "IEEE118": "pglib_opf_case118_ieee.m"}


def load_pglib(name):
    text = open(os.path.join(HERE, "data", PGLIB_FILES[name])).read()
    base = float(re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", text).group(1))
    bus, gen, br = _matrix(text, "bus"), _matrix(text, "gen"), _matrix(text, "branch")
    return dict(baseMVA=base, bus=bus, gen=gen, branch=br)
