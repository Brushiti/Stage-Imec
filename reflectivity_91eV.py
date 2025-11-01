"""
Author: Blerta Rushiti
Reflectivity Analysis for Elliptical Mirrors at 91 eV (EUV)

This script calculates and visualizes the reflectivity properties of an elliptical mirror
for various materials. It performs the following analyses:

1. Geometric calculations:
   - Defines an ellipse with semi-major axis a and semi-minor axis b
   - Computes incident angles along the upper half of the ellipse (y ≥ 0)
   
2. Angular analysis:
   - Calculates incident angles using the interior normal method
   - Converts incident angles to grazing angles for reflectivity calculations

Parameters:
    a: semi-major axis 
    b: semi-minor axis 
    
Outputs:
    - Plot of incident angles vs. x position
    - Reflectivity analysis for different materials
    - Saved figures in PNG format
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Ellipse parameters ---
a = 7
b = 1

# Foci
c = np.sqrt(a**2 - b**2)
F1 = (-c, 0)
F2 = (c, 0)

# Points on the ellipse
theta = np.linspace(0, 2*np.pi, 500)
x = a * np.cos(theta)
y = b * np.sin(theta)

angles_incident = []

# --- Incident angle calculation (interior normal) ---
for xi, yi in zip(x, y):
    if yi > 1e-12:  # only northern hemisphere
        # incident vector from F1
        vi = np.array([xi - F1[0], yi - F1[1]])
        # ellipse tangent slope: dy/dx
        slope_tangent = -(b**2 * xi) / (a**2 * yi)
        tangent = np.array([1.0, slope_tangent])
        normal = np.array([-tangent[1], tangent[0]])

        # orient the normal inward (towards center)
        if np.dot(normal, np.array([xi, yi])) > 0:
            normal = -normal

        uvi = vi / np.linalg.norm(vi)
        un = normal / np.linalg.norm(normal)
        cos_theta = np.clip(np.dot(uvi, un), -1, 1)
        theta_incident = np.arccos(cos_theta)
        if theta_incident > np.pi/2:
            theta_incident = np.pi - theta_incident
        angles_incident.append(theta_incident)

angles_incident = np.array(angles_incident)

# --- Filter points y > 0 ---
x_upper = x[y > 0]
angles_upper = angles_incident

# --- Plot incident angle ---
plt.figure(figsize=(8,5))
plt.plot(x_upper, np.degrees(angles_upper), 'b-', label="Incident angle (°)")
plt.xlabel("x position")
plt.ylabel("Incident angle (°)")
plt.title("Incident angle on the ellipse (y ≥ 0)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("IncidentAngle_91eV.png", dpi=300)
plt.show()

# --- Materials and Henke files ---
materials = {
    "Si": "Si_91eV.txt",
    "Al": "Al_91eV.txt",
    "Mo/Si": "MoSi_91eV.txt",
    "SiO2": "SiO2_91eV.txt",
    "MultiLayer_Si_Mo": "MultiLayer_Si_Mo_91eV.txt"
}

# --- Convert incident angle -> grazing angle ---
angles_inc_deg = np.degrees(angles_upper)
theta_glancing_deg = 90 - angles_inc_deg
theta_glancing_deg = np.clip(theta_glancing_deg, 0, 90)

# --- Plot reflectivity ---
plt.figure(figsize=(8,5))
for mat, filename in materials.items():
    data = np.genfromtxt(filename, skip_header=1, comments="#", usecols=(0,1))
    angles_henke = data[:,0]
    R_henke = data[:,1]

    interp_R = interp1d(angles_henke, R_henke, kind='linear',
                        bounds_error=False, fill_value=(R_henke[0], R_henke[-1]))
    
    R_points = interp_R(theta_glancing_deg)
    R_points = np.clip(R_points, 0, 1)

    plt.plot(angles_inc_deg, R_points, label=f"{mat}")

plt.xlabel("Incident angle (°) w.r.t. normal")
plt.ylabel("Reflectivity")
plt.title("Material reflectivity (E = 91 eV)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("MaterialReflectivity_91eV.png", dpi=300)
plt.show()

# --- Find highest incident angle ---
max_idx = np.argmax(angles_upper)
theta_max = np.degrees(angles_upper[max_idx])
x_max = x_upper[max_idx]
y_max = y[y>0][max_idx]
theta_glancing_max = 90 - theta_max
theta_glancing_max = np.clip(theta_glancing_max, 0, 90)

print(f"Highest incident angle: {theta_max:.2f}° at x = {x_max:.2f}, y = {y_max:.2f}")
print(f"Grazing angle at this point: {theta_glancing_max:.2f}°\n")

# --- Reflectivity at highest angle for each material ---
print("Reflectivity at highest angle:")
for mat, filename in materials.items():
    data = np.genfromtxt(filename, skip_header=1, comments="#", usecols=(0,1))
    angles_henke = data[:,0]
    R_henke = data[:,1]

    interp_R = interp1d(angles_henke, R_henke, kind='linear',
                        bounds_error=False, fill_value=(R_henke[0], R_henke[-1]))
    
    R_max = float(interp_R(theta_glancing_max))
    R_max = np.clip(R_max, 0, 1)

    print(f"{mat}: Reflectivity = {R_max:.4f}")
