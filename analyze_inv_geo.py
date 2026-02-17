from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
outdir = BASE_DIR / "output"

trace_file = outdir / "trace.txt"
bmodel_file = BASE_DIR / "bmodel_file.py"
diag_file = outdir / "diagnostics.txt"

# =========================================================
# PHYSICAL CONSTANTS (must match simulation)
# =========================================================

q = -1.602e-19      # electron charge
m = 9.109e-31       # electron mass

# =========================================================
# LOAD ANALYTIC MAGNETIC FIELD MODEL
# =========================================================

spec = importlib.util.spec_from_file_location("bmodel", bmodel_file)
bmodel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bmodel)

EvalB = bmodel.B_model

# =========================================================
# NUMERICAL PARAMETERS
# =========================================================

# spatial step for finite differences (adjust if needed)
EPS = 1e-5


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def pitch_angle(v, B):
    Bmag = np.linalg.norm(B)
    vmag = np.linalg.norm(v)
    if Bmag == 0 or vmag == 0:
        return np.nan
    vpar = np.dot(v, B) / Bmag
    return np.arccos(np.clip(vpar / vmag, -1, 1)) # clip for numerical stability


def magnetic_moment(v, B):
    Bmag = np.linalg.norm(B)
    if Bmag == 0:
        return np.nan
    vpar = np.dot(v, B) / Bmag
    vperp2 = np.dot(v, v) - vpar**2
    return m * vperp2 / (2 * Bmag) # magnetic moment mu = (1/2) m v_perp^2 / B


def larmor_radius(v, B):
    Bmag = np.linalg.norm(B)
    if Bmag == 0:
        return np.nan
    vpar = np.dot(v, B) / Bmag
    vperp = np.sqrt(max(np.dot(v, v) - vpar**2, 0)) 
    return m * vperp / (abs(q) * Bmag) # Larmor radius rho_L = m v_perp / (|q| B)


# ---------------------------------------------------------
# FIELD CURVATURE
# ---------------------------------------------------------

def unit_B(pos):
    B = EvalB(pos)
    Bmag = np.linalg.norm(B)
    if Bmag == 0:
        return np.zeros(3)
    return B / Bmag # unit vector of B


def grad_unit_B(pos):
    """
    Compute gradient of unit magnetic field using central differences.
    Returns 3x3 tensor: d(b_i)/dx_j
    """
    grad = np.zeros((3, 3))

    for j in range(3):
        d = np.zeros(3)
        d[j] = EPS

        b_plus = unit_B(pos + d)
        b_minus = unit_B(pos - d)

        grad[:, j] = (b_plus - b_minus) / (2 * EPS)

    return grad


def curvature_radius(pos):
    """
    Rc = 1 / | (b · ∇) b |
    """
    b = unit_B(pos)
    gradb = grad_unit_B(pos)

    kappa_vec = gradb @ b   # directional derivative
    kappa = np.linalg.norm(kappa_vec)

    if kappa == 0:
        return np.inf

    return 1.0 / kappa


# =========================================================
# LOAD TRAJECTORY
# =========================================================

df = pd.read_csv(trace_file, sep=r"\s+")

time = df["time"].values
positions = df[["x", "y", "z"]].values
velocities = df[["u", "v", "w"]].values


# =========================================================
# COMPUTE DIAGNOSTICS
# =========================================================

pitch_list = []
mu_list = []
rhoL_list = []
Rc_list = []

print("Computing diagnostics...")

for pos, vel in zip(positions, velocities):

    B = EvalB(pos)

    pitch_list.append(pitch_angle(vel, B))
    mu_list.append(magnetic_moment(vel, B))
    rhoL_list.append(larmor_radius(vel, B))
    Rc_list.append(curvature_radius(pos))


pitch_arr = np.array(pitch_list)
mu_arr = np.array(mu_list)
rhoL_arr = np.array(rhoL_list)
Rc_arr = np.array(Rc_list)


# =========================================================
# SAVE OUTPUT FILE
# =========================================================

with open(diag_file, "w") as f:
    f.write("time x y z pitch mu rL Rc\n")

    for i in range(len(time)):
        f.write(
            f"{time[i]:g} "
            f"{positions[i,0]:g} "
            f"{positions[i,1]:g} "
            f"{positions[i,2]:g} "
            f"{pitch_arr[i]:g} "
            f"{mu_arr[i]:g} "
            f"{rhoL_arr[i]:g} "
            f"{Rc_arr[i]:g}\n"
        )

print("Saved:", diag_file)
print("Done.")
