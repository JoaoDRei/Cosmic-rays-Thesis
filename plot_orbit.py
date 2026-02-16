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

trace_file = outdir / "trace.txt"
Bfile = outdir / "Bfield_yz.txt"

# --------------------------------
# Load particle data
# --------------------------------
df = pd.read_csv(trace_file, sep=r"\s+")

# --------------------------------
# Load magnetic field snapshot
# --------------------------------
# --------------------------------
# Load magnetic field snapshot
# --------------------------------
dfB = pd.read_csv(Bfile, sep=r"\s+")

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




# --------------------------------
# 2D gyro-orbit (x-y plane)
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
# 2D orbit (z-y plane)
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
# Magnetic field lines (y-z plane)
# --------------------------------
plt.figure()
plt.streamplot(Z, Y, Bz, By, color=Bmag/Bmag.max(), cmap="plasma", density=1.2)

plt.colorbar(label="|B| [arb. units]")

plt.xlabel("z [m]")
plt.ylabel("y [m]")
plt.title("Magnetic field lines (y-z plane)")
plt.grid(True)

Bfield_path = outdir / "Bfield_yz.png"
plt.savefig(Bfield_path, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------
# 3D trajectory
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
# Report saved files
# --------------------------------
print("Saved figures:")
print(xy_path)
print(zy_path)
print(Bfield_path)
print(xyz_path)
