import code
import time
import math
import os
from pathlib import Path
from turtle import pos
import numpy as np
BASE_DIR = Path(__file__).resolve().parent
outdir = BASE_DIR / "output"
outdir.mkdir(exist_ok=True)

#git add .
#git commit -m "Describe what changed"
#git push


# ----------------------------
# Particle class
# ----------------------------
class Particle:
    def __init__(self):
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.q = -1.602e-19   # electron charge
        self.m = 9.109e-31    # electron mass
    
    def pitch_angle(self, B):
        B_mag = np.linalg.norm(B) #Magnitude of B using Euclidian norm
        v_mag = np.linalg.norm(self.v) #Magnitude of velocity using Euclidian norm
        v_par = np.dot(self.v, B) / B_mag #Paralell velocity component
        return np.arccos(v_par / v_mag) #pitch angle

    def magnetic_moment(self, B):
        B_mag = np.linalg.norm(B)
        v_mag = np.linalg.norm(self.v)
        v_par = np.dot(self.v, B) / B_mag
        v_perp_sq = v_mag**2 - v_par**2
        return self.m * v_perp_sq / (2 * B_mag) #magnetic moment using the formula mu = m * v_perp^2 / (2 * B)

    def larmor_radius(self,B):
        B_mag = np.linalg.norm(B)
        v_mag= np.linalg.norm(self.v)
        v_par = np.dot(self.v, B) / B_mag #parallel velocity component
        v_perp = np.sqrt(v_mag**2 - v_par**2) #perpendicular velocity component using the relation v^2 = v_par^2 + v_perp^2
        return self.m * v_perp / (abs(self.q) * B_mag) #Larmor radius using the formula rL = m * v_perp / (|q| * B)
# ----------------------------
# Field evaluators
# ----------------------------
def EvalE(pos):
    return [0.0, 0.0, 0.0] #will be kept at zero
def EvalB(pos):
    return B_MODEL_FUNC(pos)   
#Defining the different types of Magnetic Field:
def ConstantzB(pos):
    return [0.0, 0.0, 0.01]
def MirrorStraightGradB(pos):
    B0 = 1.0
    L = 10.0
    return np.array([0.0, - B0 * (1 + pos[2]*pos[2]/L**2),0]) 
def MirrorHourglassB(pos):
    B0 = 1.0
    L = 10.0
    return np.array([B0*(-pos[0]/L), B0*(-pos[1]/L), B0 * (1 + (pos[2]/L)**2)])
def MirrorFiniteBottle(pos):
    B0 = 1.0
    L = 10.0
    return np.array([0.0, 0.0, B0 * (1 + np.tanh(pos[2]/L)**2)])
def TwistTanh(pos):
    B0 = 1.0
    By0 = 1
    Bz0 = 1.0
    L = 1
    return np.array([0, By0 , Bz0* np.tanh(pos[1]/L)])

#B-field model selection
FIELD_MODELS = {
    "const": ConstantzB,
    "straightGrad": MirrorStraightGradB,
    "hourglass": MirrorHourglassB,
    "finiteBottle": MirrorFiniteBottle,
    "TwistTanh": TwistTanh,
}
B_MODEL_NAME = "TwistTan"
B_MODEL_FUNC = FIELD_MODELS[B_MODEL_NAME]


#Write the B model function to a file for analysis later
import inspect

def write_bmodel_file(field_function):

    func_source = inspect.getsource(field_function)

    code = f"""import numpy as np

{func_source}

def B_model(pos):
    return {field_function.__name__}(pos)
"""

    (BASE_DIR / "bmodel_file.py").write_text(code)
#----------------------------
# Save magnetic field snapshot for plotting. Just the z-y plane is saved
#----------------------------
def SaveMagneticFieldSnapshot(x_plane):
    z_vals = np.linspace(-2.0, 2.0, 80)
    y_vals = np.linspace(-2.0, 2.0, 80)

    with open(outdir / "Bfield_yz.txt", "w") as f:
        f.write("z y Bx By Bz\n")

        for z in z_vals:
            for y in y_vals:
                B = EvalB([x_plane, y, z])   # x = 0 plane
                f.write(f"{z:g} {y:g} {B[0]:g} {B[1]:g} {B[2]:g}\n")
# ----------------------------
# Vector utilities
# ----------------------------
def CrossProduct(v1, v2):
    return [
        v1[1]*v2[2] - v1[2]*v2[1],
        -v1[0]*v2[2] + v1[2]*v2[0],
        v1[0]*v2[1] - v1[1]*v2[0]
    ]
def Determinant(a):
    return (
        a[0][0]*(a[1][1]*a[2][2] - a[1][2]*a[2][1])
        - a[0][1]*(a[1][0]*a[2][2] - a[1][2]*a[2][0])
        + a[0][2]*(a[1][0]*a[2][1] - a[1][1]*a[2][0])
    )
def MatrixVectMult(a, x):
    return [
        a[0][0]*x[0] + a[0][1]*x[1] + a[0][2]*x[2],
        a[1][0]*x[0] + a[1][1]*x[1] + a[1][2]*x[2],
        a[2][0]*x[0] + a[2][1]*x[1] + a[2][2]*x[2],
    ]
def VectVectAdd(a, b):
    return [a[i] + b[i] for i in range(3)]
# ----------------------------
# Particle pusher: updates position based on velocity
# ----------------------------
def PushParticle(part, dt):
    for i in range(3):
        part.x[i] += part.v[i] * dt
# ----------------------------
# Velocity update methods
# ----------------------------
def UpdateVelocity(part, E, B, dt):
    # Choose method here
    UpdateVelocityBoris(part, E, B, dt)

def UpdateVelocityForward(part, E, B, dt):
    vxB = CrossProduct(part.v, B)
    for i in range(3):
        part.v[i] += (part.q/part.m) * (E[i] + vxB[i]) * dt
def UpdateVelocityBoris(part, E, B, dt):
    #defining intermediate velocities: v-, v', v+, where v' is the velocity after the half electric field acceleration, and v+ is the velocity after the magnetic rotation
    v_minus = [0.0]*3 
    v_prime = [0.0]*3
    v_plus = [0.0]*3

    t = [(part.q/part.m) * B[i] * 0.5 * dt for i in range(3)] # t is the normalized magnetic field vector scaled by the time step and charge-to-mass ratio t=(q/m)*B*dt/2
    t_mag2 = t[0]**2 + t[1]**2 + t[2]**2 # magnitude squared of t
    s = [2*t[i] / (1 + t_mag2) for i in range(3)] # s is the vector used for the magnetic rotation, calculated as s = 2*t / (1 + |t|^2)

    # v-
    for i in range(3):
        v_minus[i] = part.v[i] + (part.q/part.m) * E[i] * 0.5 * dt #defining v- as the velocity after the half electric field acceleration, calculated as v- = v + (q/m)*E*dt/2. In this case, the electric field is zero, so v- is equal to the initial velocity v.

    # v'
    v_minus_cross_t = CrossProduct(v_minus, t)
    for i in range(3):
        v_prime[i] = v_minus[i] + v_minus_cross_t[i] # v' is the velocity after half the magnetic rotation


    # v+
    v_prime_cross_s = CrossProduct(v_prime, s)
    for i in range(3):
        v_plus[i] = v_minus[i] + v_prime_cross_s[i] # v+ is the velocity after the full magnetic rotation,

    # final velocity
    for i in range(3):
        part.v[i] = v_plus[i] + (part.q/part.m) * E[i] * 0.5 * dt #the final velocity. in this case, since the electric field is zero, the final velocity is just v+.
# ----------------------------
# Sample particles
# ----------------------------
def SampleParticleLr():
    part = Particle() #initiates a particle with zero position and velocity, and the charge and mass of an electron
    
    part.v[1] = 0.1 # sets the y-component of the velocity to 100,000 m/s
    part.v[2] = 1e4 # sets the z-component of the velocity to 10,000 m/s, which means that the particle has a small velocity component along the magnetic field direction (z-axis) in addition to its larger velocity component perpendicular to the magnetic field (y-axis). This setup allows us to observe both the gyration around the magnetic field lines and the motion along the field lines.
    B = EvalB(part.x)  # evaluates the magnetic field at the particle's position
    rL= part.larmor_radius(B)
    #rL2 = part.m * part.v[1] / (abs(part.q) * B[2]) # calculates the Larmor radius
    part.x[0] = rL # sets the x-component of the position to the Larmor radius, which means that the particle starts at a distance from the origin equal to the Larmor radius along the x-axis. This is a common choice for initializing a particle in a magnetic field, as it allows us to observe the circular motion of the particle around the magnetic field lines.

    print(f"Larmor radius is {rL:g}")
    #print(f"Original Larmor radius is {rL2:g}")
    return part
def SampleParticle2():
    part = Particle() #initiates a particle with zero position and velocity, and the charge and mass of an electron
    part.x[1]=0.1
    part.v[1] = 0.1 # sets the y-component of the velocity to 0.1 m/s
    part.v[2] = 1e4 # sets the z-component of the velocity to 10,000 m/s, which means that the particle has a small velocity component along the magnetic field direction (z-axis) in addition to its larger velocity component perpendicular to the magnetic field (y-axis). This setup allows us to observe both the gyration around the magnetic field lines and the motion along the field lines.
    B = EvalB(part.x)  # evaluates the magnetic field at the particle's position
    #rL = part.m * part.v[1] / (abs(part.q) * B[2]) # calculates the Larmor radius
    rL=part.larmor_radius(B)
    part.x[0] = rL # sets the x-component of the position to the Larmor radius, which means that the particle starts at a distance from the origin equal to the Larmor radius along the x-axis. This is a common choice for initializing a particle in a magnetic field, as it allows us to observe the circular motion of the particle around the magnetic field lines.

    #print(f"Larmor radius is {rL:g}")
    return part
# ----------------------------
# Main program
# ----------------------------
def main():
    dt = 3e-11
    it_max = 1000 # sets the time step (dt) to 3e-11 seconds and the maximum number of iterations (it_max) to 1000. This means that the simulation will run for a total time of it_max * dt = 3e-8 seconds, which should be sufficient to observe several gyrations of the particle around the magnetic field lines, given the initial velocity and magnetic field strength.
    write_bmodel_file(B_MODEL_FUNC)
    with open(outdir / "trace.txt", "w") as f: # opens a file named "trace.txt" in write mode. This file will be used to store the output of the simulation, including the time step, time, position, and velocity of the particle at each recorded step.
        f.write("it time x y z u v w Bx By Bz\n")

        # initializes a particle with specific initial conditions.
        part = SampleParticle2()  #SampleParticleLr(), SampleParticle2() 

        #Magnetic Field Snapshot at particle's initial x plane
        SaveMagneticFieldSnapshot(part.x[0]) 

        # Evaluates the electric and magnetic fields at the particle's position and pushes the velocity back by half step
        E = EvalE(part.x)
        B = EvalB(part.x)


        UpdateVelocity(part, E, B, -0.5*dt)

        start_time = time.time()

        for it in range(it_max):
            # fields at current position
            E = EvalE(part.x)
            B = EvalB(part.x)

            # advance particle
            UpdateVelocity(part, E, B, dt)
            PushParticle(part, dt)

            # diagnostics at new state
            B = EvalB(part.x)
            #pitch = part.pitch_angle(B)
            #mu = part.magnetic_moment(B)    

            SAVE_EVERY = 2
            if it % SAVE_EVERY == 0: # records the particle's state every SAVE_EVERY iterations (every SAVE_EVERY*3e-11 seconds) by writing the current iteration number, time, position, and velocity to the "trace.txt" file. This allows us to track the particle's trajectory and velocity over time, which can be useful for analyzing its motion in the magnetic field.
                f.write(
                    f"{it} {it*dt:g} "
                    f"{part.x[0]:g} {part.x[1]:g} {part.x[2]:g} "
                    f"{part.v[0]:g} {part.v[1]:g} {part.v[2]:g} "
                    f"{B[0]:g} {B[1]:g} {B[2]:g}\n"
                )

        end_time = time.time()

    print(f"Finished after {it_max} time steps in {end_time - start_time:g} seconds")
if __name__ == "__main__":
    main()
