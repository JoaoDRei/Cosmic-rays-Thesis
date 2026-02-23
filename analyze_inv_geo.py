from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util
from scipy.integrate import quad
# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
outdir = BASE_DIR / "output"/"txt"

trace_file = outdir / "trace.txt"
bmodel_file = BASE_DIR / "bmodel_file.py"
diag_file = outdir / "diagnostics.txt"
curv_and_pitch_file = outdir / "curvature_pitch.txt"
# =========================================================
# PHYSICAL CONSTANTS (must match simulation)
# =========================================================

q = -1.602e-19      # electron charge
m = 9.109e-31       # electron mass

qoverm= 1 
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
    #return m * vperp / (abs(q) * Bmag) # Larmor radius rho_L = m v_perp / (|q| B)
    return  vperp / (abs(qoverm) * Bmag)

def delta_mu_over_mu(rL, Rc,pitcheq, gyroeq):
    def Fint(x, alpha):
        return (1 - x**2) / np.sqrt(1 - (np.sin(alpha)**2) * np.sqrt(1 - x**2))
    def I(alpha):
        result, error = quad(Fint, 0.0, 1.0, args=(alpha,))
        return result
    gamma_98=0.9417426
    epsilon= rL/Rc
    Falpha= I(pitcheq)
    f1=np.pi/(np.power(2, 0.25)*gamma_98)
    f2= 1/(np.power(epsilon,1/8)*np.sin(pitcheq))
    f3= np.exp(-Falpha/epsilon)
    print(f1, f2, f3)
    print(gyroeq)
    print(Falpha/epsilon)
    return -f1*f2*f3*np.cos(gyroeq)
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

def gyrophase(v, B):
    """
    Gyrophase = angle of perpendicular velocity
    in plane perpendicular to local B.
    """

    Bmag = np.linalg.norm(B)
    if Bmag == 0:
        return np.nan

    # unit field direction
    b = B / Bmag

    # perpendicular velocity
    v_par = np.dot(v, b)
    v_perp = v - v_par * b
    vperp_mag = np.linalg.norm(v_perp)

    if vperp_mag == 0:
        return np.nan

    # build perpendicular orthonormal basis, i.e., a basis for the plane perpendicular to b. this basis is composed by two unit vectors e1 and e2, which are orthogonal to each other and to b. we can construct e1 by taking the cross product of b with an arbitrary reference vector (e.g., [1,0,0]) that is not parallel to b. then we normalize e1 to make it a unit vector. finally, we can get e2 by taking the cross product of b with e1, which will also be a unit vector and orthogonal to both b and e1.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, b)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(b, ref)
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(b, e1)

    # components in perpendicular plane
    v1 = np.dot(v_perp, e1)
    v2 = np.dot(v_perp, e2)

    return np.arctan2(v2, v1) + np.pi # shift to [0, 2pi]
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
gyro_list = []
print("Computing diagnostics...")

for pos, vel in zip(positions, velocities):

    B = EvalB(pos)

    pitch_list.append(pitch_angle(vel, B))
    mu_list.append(magnetic_moment(vel, B))
    rhoL_list.append(larmor_radius(vel, B))
    Rc_list.append(curvature_radius(pos))
    gyro_list.append(gyrophase(vel, B))

pitch_arr = np.array(pitch_list)
mu_arr = np.array(mu_list)
rhoL_arr = np.array(rhoL_list)
Rc_arr = np.array(Rc_list)
gyro_arr = np.array(gyro_list)

# ====================================================================
#Maximum values for curvatures (min Rc) and corresponding pitch angles
# ====================================================================
local_min_Rc_index = np.where((Rc_arr[1:-1] < Rc_arr[:-2]) & (Rc_arr[1:-1] < Rc_arr[2:]))[0] + 1
local_min_Rc_values = Rc_arr[local_min_Rc_index]#curvature radius at the bends
local_min_pitch_values = pitch_arr[local_min_Rc_index] #pitch angles at the bends
gyrophase_at_bends = gyro_arr[local_min_Rc_index] #gyrophase at the bends
rL_at_bends = rhoL_arr[local_min_Rc_index] #Larmor radius at the bends


delta_mu_over_mu_values=[]
for rL, Rc, pitch, gyro in zip(rL_at_bends, local_min_Rc_values, local_min_pitch_values, gyrophase_at_bends):
    
    delta_mu_over_mu_values.append(delta_mu_over_mu(rL, Rc, pitch, gyro))



# =========================================================
# SAVE OUTPUT FILE
# =========================================================

with open(diag_file, "w") as f:
    f.write("time x y z pitch mu rL Rc gyrophase\n")

    for i in range(len(time)):
        f.write(
            f"{time[i]:g} "
            f"{positions[i,0]:g} "
            f"{positions[i,1]:g} "
            f"{positions[i,2]:g} "
            f"{pitch_arr[i]:g} "
            f"{mu_arr[i]:g} "
            f"{rhoL_arr[i]:g} "
            f"{Rc_arr[i]:g} "
            f"{gyro_arr[i]:g}\n"    
        )


with open(curv_and_pitch_file, "w") as f:
    f.write("timeindex minRadius equatorialPitch deltaMuOverMu\n")

    for i in range(len(local_min_Rc_values)):
        f.write(
            f"{local_min_Rc_index[i]} "
            f"{local_min_Rc_values[i]:g} "
            f"{local_min_pitch_values[i]:g} "
            f"{delta_mu_over_mu_values[i]:g}\n"
        )



print("Saved:", diag_file)
print("Saved:", curv_and_pitch_file)
print("Done.")
