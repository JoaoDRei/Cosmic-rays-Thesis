from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from matplotlib.ticker import ScalarFormatter

# scientific notation formatter
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))   # always use scientific notation

# --------------------------------
# Paths
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent
outdir = BASE_DIR / "output"
outdirtxt = outdir / "txt"
trace_file = outdirtxt / "trace.txt"
Bfile = outdirtxt / "Bfield_yz.txt"
diagnostics_file = outdirtxt / "diagnostics.txt"
curv_and_pitch_file = outdirtxt / "curvature_pitch.txt"
# --------------------------------
# Load particle data
# --------------------------------
df = pd.read_csv(trace_file, sep=r"\s+")
dfB = pd.read_csv(Bfile, sep=r"\s+")
df_diag = pd.read_csv(diagnostics_file, sep=r"\s+")

# --------------------------------
# Trace: 2D gyro-orbit (x-y plane)
# --------------------------------
plt.figure()
plt.plot(df["x"], df["y"])
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Gyromotion xy-plane")
plt.axis("equal")
plt.grid(True)

xy_path = outdir / "orbit_xy.png"
plt.savefig(xy_path, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------
# Trace:2D orbit (z-y plane)
# --------------------------------
plt.figure()
plt.plot(df["z"], df["y"])
plt.xlabel("z [m]")
plt.ylabel("y [m]")
plt.title("Motion in z-y plane")
plt.grid(True)

zy_path = outdir / "orbit_zy.png"
plt.savefig(zy_path, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------
# Trace:2D orbit (x-z plane)
# --------------------------------
plt.figure()
plt.plot(df["x"], df["z"])
plt.xlabel("x [m]")
plt.ylabel("z [m]")
plt.title("Motion in x-z plane")
plt.grid(True)

xz_path = outdir / "orbit_xz.png"
plt.savefig(xz_path, dpi=300, bbox_inches="tight")
plt.close()




# --------------------------------
# Trace:Magnetic field lines (y-z plane) quiver plot
# --------------------------------
# number of grid points (must match simulation snapshot)
NZ = len(np.unique(dfB["z"]))
NY = len(np.unique(dfB["y"]))

# rebuild uniform coordinate arrays directly
z_vals = np.linspace(dfB["z"].min(), dfB["z"].max(), NZ)
y_vals = np.linspace(dfB["y"].min(), dfB["y"].max(), NY)

Z, Y = np.meshgrid(z_vals, y_vals)

# reshape field components in correct order
Bz = dfB["Bz"].values.reshape(NZ, NY).T
By = dfB["By"].values.reshape(NZ, NY).T
# compute magnitude
Bmag = np.sqrt(Bz**2 + By**2)
skip = 4  # take every 4th point
plt.figure()
plt.quiver(
    Z[::skip, ::skip], Y[::skip, ::skip],
    Bz[::skip, ::skip], By[::skip, ::skip],
    Bmag[::skip, ::skip],
    angles="xy", scale_units="xy", scale=5, cmap="plasma"
)
plt.colorbar(label="|B| [arb. units]")
plt.xlabel("z [m]")
plt.ylabel("y [m]")
plt.title("Magnetic field vectors (y-z plane)")
plt.grid(True)
Bfield_path_quiver = outdir / "Bfield_yz_quiver.png"
plt.savefig(Bfield_path_quiver, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------
# Trace:3D trajectory
# --------------------------------
fig = plt.figure(figsize=(20, 6), constrained_layout=True)

ax = fig.add_subplot(111, projection="3d")
ax.plot(df["x"], df["y"], df["z"])
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("3D particle trajectory")

# apply to all axes
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
ax.zaxis.set_major_formatter(formatter)




xyz_path = outdir / "orbit_3d.png"
plt.savefig(xyz_path, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------
# Diagnostics: pitch angle, magnetic moment, Larmor radius, curvature radius
# --------------------------------

# constants (same as simulation)
q = 1.602e-19
m = 9.109e-31

time = df["time"].values

pos = df[["x","y","z"]].values
vel = df[["u","v","w"]].values
Btraj = df[["Bx","By","Bz"]].values
pitch = df_diag["pitch"].values
mu = df_diag["mu"].values
rL= df_diag["rL"].values
Rc = df_diag["Rc"].values
gyrophase= df_diag["gyrophase"].values

# --------------------------------
# statistics for pitch angle and magnetic moment
# --------------------------------

def summarize(name, arr):
    mean = np.mean(arr)
    std = np.std(arr)
    rel_std = std / abs(mean) if mean != 0 else np.nan
    drift = (arr[-1] - arr[0]) / abs(arr[0]) if arr[0] != 0 else np.nan

    print(f"{name}:")
    print(f"   mean        = {mean:.6e}")
    print(f"   std         = {std:.6e}")
    print(f"   rel std     = {rel_std:.6e}")
    print(f"   total drift = {drift:.6e}")
    print()

summarize("Pitch angle", pitch)
summarize("Magnetic moment", mu)

# --------------------------------
# time series plots: pitch angle, magnetic moment and gyrophase
# --------------------------------

plt.figure()
plt.plot(time, pitch)
plt.xlabel("time [s]")
plt.ylabel("pitch angle [rad]")
plt.title("Pitch angle vs time")
plt.grid(True)
pitch_path = outdir / "pitch_timeseries.png"
plt.savefig(pitch_path, dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(time, mu)
plt.xlabel("time [s]")
plt.ylabel("magnetic moment")
plt.title("Magnetic moment vs time")
plt.grid(True)
mu_path = outdir / "mu_timeseries.png"
plt.savefig(mu_path, dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(time, gyrophase)
plt.xlabel("time [s]")
plt.ylabel("gyrophase [rad]")
plt.title("Gyrophase vs time")
plt.grid(True)
gyro_path = outdir / "gyro_timeseries.png"
plt.savefig(gyro_path, dpi=300, bbox_inches="tight")
plt.close() 

# --------------------------------
# conservation error
# --------------------------------

pitch_rel = (pitch - pitch[0]) / pitch[0]
mu_rel = (mu - mu[0]) / mu[0]

plt.figure()
plt.plot(time, pitch_rel)
plt.xlabel("time [s]")
plt.ylabel("relative change")
plt.title("Pitch angle conservation error")
plt.grid(True)
pitch_err_path = outdir / "pitch_relative_error.png"
plt.savefig(pitch_err_path, dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(time, mu_rel)
plt.xlabel("time [s]")
plt.ylabel("relative change")
plt.title("Magnetic moment conservation error")
plt.grid(True)
mu_err_path = outdir / "mu_relative_error.png"
plt.savefig(mu_err_path, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------
# curvature vs Larmor radius comparison
# --------------------------------

plt.figure()
plt.plot(time, rL, label="Larmor radius")
plt.plot(time, Rc, label="Curvature radius")
plt.xlabel("time [s]")
plt.ylabel("length [m]")
plt.title("Larmor radius vs magnetic curvature radius")
plt.legend()
plt.grid(True)
curv_compare_path = outdir / "rho_vs_Rc.png"
plt.savefig(curv_compare_path, dpi=300, bbox_inches="tight")
plt.close()

# ratio diagnostic 
ratio = rL / Rc

plt.figure()
plt.plot(time, ratio)
plt.xlabel("time [s]")
plt.ylabel("rho_L / Rc")
plt.title("Adiabatic validity parameter")
plt.grid(True)
ratio_path = outdir / "rho_over_Rc.png"
plt.savefig(ratio_path, dpi=300, bbox_inches="tight")
plt.close()

#=====================================
#Radius of curvature time series
#=====================================

curv_pitch_df = pd.read_csv(curv_and_pitch_file, sep=r"\s+")
min_indices = curv_pitch_df["timeindex"].values.astype(int)


plt.figure()
plt.plot(time, Rc, label="Curvature radius")

plt.scatter(
    time[min_indices],
    Rc[min_indices],
    zorder=3,
    label="Local minima",
    marker="o"
)

plt.xlabel("time [s]")
plt.ylabel("Curvature radius [m]")
plt.title("Curvature radius with local minima")
plt.legend()
plt.grid(True)

Rc_min_path = outdir / "Rc_timeseries_with_minima.png"
plt.savefig(Rc_min_path, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------
# Report saved files
# --------------------------------
print("Saved figures:")
print(xy_path)
print(zy_path)
print(xz_path)
print(Bfield_path_quiver)
print(xyz_path)
print(pitch_path)
print(mu_path)
print(gyro_path)
print(pitch_err_path)
print(mu_err_path)
