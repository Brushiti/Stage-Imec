"""
Author: Blerta Rushiti

Reflectivity Analysis for Elliptical Mirrors - Parameter Study

This script analyzes how the reflectivity of an elliptical mirror varies with 
its geometric parameters, specifically studying the impact of changing the 
semi-major axis 'a' while keeping 'b' constant.

Features:
1. Parameter sweep:
   - Fixed semi-minor axis b = 10
   - Variable semi-major axis a from 250 to 2000
   - Analysis for multiple materials

2. Calculations for each configuration:
   - Best achievable reflectivity
   - Worst reflectivity
   - Percentage of surface with R ≥ 0.9
   - Angular spread between extreme rays

3. Visualizations:
   - Maximum reflectivity vs. semi-major axis
   - Percentage of high-reflectivity surface vs. a
   - Angle between extreme rays vs. a

Input files:
    - Henke data files for each material at 350 eV

Parameters:
    b: semi-minor axis (fixed at 10)
    a_values: array of semi-major axis values to analyze
    threshold: minimum reflectivity considered "good" (0.9)

"""

# ...existing code...
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

b = 10
#a_values = [25, 35, 45, 55, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
#a_values = [50, 70, 90, 110, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]
a_values = [250,350,450,550,650,700,750,800,850,900,950,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
materials = {
    #"Al": "Al_350eV.txt",
    #"SiO2": "SiO2_350eV.txt",
    "Ni": "Ni_350eV.txt",
    "Cr/Ti": "Cr-Ti_350eV.txt",
    "Cr/V": "Cr-V_350eV.txt"#,
    #"Cr/Sc": "Cr-Sc_350eV.txt",
    #"W/B4C": "W-B4C_350eV.txt",
    #"Mo/Si": "Mo-Si_350eV.txt"
}

best_ref = {m: [] for m in materials.keys()}
worst_ref = {m: [] for m in materials.keys()}
percent_high = {m: [] for m in materials.keys()}  # pourcentage d'arc avec R >= seuil
angles_between = {m: [] for m in materials.keys()}  # angle entre les deux droites demandées (deg)

threshold = 0.9  # réfléctivité minimale

for a in a_values:
    # ellipse and source
    c = np.sqrt(max(a**2 - b**2, 0.0))
    F1 = (-c, 0.0)

    theta = np.linspace(0, 2*np.pi, 4000)
    x = a * np.cos(theta)
    y = b * np.sin(theta)

    # select lower half (y < 0) and keep order
    mask_lower = y < 0
    x_lower = x[mask_lower]
    y_lower = y[mask_lower]
    if x_lower.size < 2:
        # pas assez de points
        for m in materials:
            best_ref[m].append(0.0)
            worst_ref[m].append(0.0)
            percent_high[m].append(0.0)
        continue

    # compute incident angles (degrees) for lower half points
    angles_incident_deg = []
    for xi, yi in zip(x_lower, y_lower):
        vi = np.array([xi - F1[0], yi - F1[1]])
        norm_vi = np.linalg.norm(vi)
        if norm_vi == 0:
            angles_incident_deg.append(0.0)
            continue
        slope_tangent = -(b**2 * xi) / (a**2 * yi)
        tangent = np.array([1.0, slope_tangent])
        normal = np.array([-tangent[1], tangent[0]])
        # orient normal inward (vers le centre)
        if np.dot(normal, np.array([xi, yi])) > 0:
            normal = -normal
        uvi = vi / norm_vi
        un = normal / np.linalg.norm(normal)
        cos_theta = np.clip(np.dot(uvi, un), -1.0, 1.0)
        theta_inc = np.arccos(cos_theta)
        if theta_inc > np.pi/2:
            theta_inc = np.pi - theta_inc
        angles_incident_deg.append(np.degrees(theta_inc))
    angles_incident_deg = np.array(angles_incident_deg)

    theta_glancing_deg = np.clip(90.0 - angles_incident_deg, 0.0, 90.0)

    # arc lengths along the lower half (approximation by piecewise linear segments)
    dx = np.diff(x_lower)
    dy = np.diff(y_lower)
    seg_lengths = np.sqrt(dx**2 + dy**2)
    total_length = np.sum(seg_lengths)
    if total_length == 0:
        for m in materials:
            best_ref[m].append(0.0)
            worst_ref[m].append(0.0)
            percent_high[m].append(0.0)
        continue

    # for each material: interpolate reflectivity, compute best/worst and percent of arc >= threshold
    for mat, fname in materials.items():
        if not os.path.isfile(fname):
            print(f"Warning: file '{fname}' not found. Setting values to 0 for a={a}, material={mat}.")
            best_ref[mat].append(0.0)
            worst_ref[mat].append(0.0)
            percent_high[mat].append(0.0)
            angles_between[mat].append(np.nan)
            continue
        data = np.genfromtxt(fname, comments="#", usecols=(0,1))
        if data.ndim != 2 or data.shape[0] < 2:
            print(f"Warning: file '{fname}' seems invalide. Setting values to 0 for a={a}, material={mat}.")
            best_ref[mat].append(0.0)
            worst_ref[mat].append(0.0)
            percent_high[mat].append(0.0)
            angles_between[mat].append(np.nan)
            continue
        angles_h = data[:,0]
        R_h = data[:,1]
        interp_R = interp1d(angles_h, R_h, kind='linear', bounds_error=False,
                            fill_value=(R_h[0], R_h[-1]))
        R_points = np.clip(interp_R(theta_glancing_deg), 0.0, 1.0)

        best_ref_val = float(np.max(R_points))
        worst_ref_val = float(np.min(R_points))
        best_ref[mat].append(best_ref_val)
        worst_ref[mat].append(worst_ref_val)

        # approximer la fraction d'arc où R >= threshold
        # on évalue la réflectivité au milieu de chaque segment comme (R_i + R_{i+1})/2
        R_mid = 0.5 * (R_points[:-1] + R_points[1:])
        good_length = np.sum(seg_lengths[R_mid >= threshold])
        percent = 100.0 * (good_length / total_length)
        percent_high[mat].append(percent)

        # --- Nouveau : angle entre deux droites partant de F1 vers les points R >= threshold
        idx_good = np.where(R_points >= threshold)[0]
        if idx_good.size == 0:
            angles_between[mat].append(np.nan)
        else:
            # indices sur x_lower / y_lower correspondent à R_points indices
            # trouver index with maximal x and minimal x among good points
            i_max = idx_good[np.argmax(x_lower[idx_good])]
            i_min = idx_good[np.argmin(x_lower[idx_good])]

            p_max = np.array([x_lower[i_max], y_lower[i_max]])
            p_min = np.array([x_lower[i_min], y_lower[i_min]])
            F1_vec = np.array([F1[0], F1[1]])

            v1 = p_max - F1_vec
            v2 = p_min - F1_vec
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                angles_between[mat].append(np.nan)
            else:
                cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angle_deg = float(np.degrees(np.arccos(cosang)))
                angles_between[mat].append(angle_deg)

# affichage des résultats
print("a values:", a_values)
for mat in materials.keys():
    print(f"{mat} best reflectivities:", ["{:.6f}".format(v) for v in best_ref[mat]])
    print(f"{mat} worst reflectivities:", ["{:.6f}".format(v) for v in worst_ref[mat]])
    print(f"{mat} percent of lower half with R >= {threshold}:",
          ["{:.2f}%".format(v) for v in percent_high[mat]])
    print(f"{mat} angle between max-x and min-x good points (deg):",
          [("{:.2f}".format(v) if (not np.isnan(v)) else "nan") for v in angles_between[mat]])

# --- Plot: best reflectivity vs a (requested) ---
plt.figure(figsize=(8,5))
for mat, values in best_ref.items():
    plt.plot(a_values, values, marker='o', label=f"{mat} (max R on lower half)")
plt.xlabel("a (ellipse semi-major axis)")
plt.ylabel("Maximum reflectivity (lower half)")
plt.title("Best reflectivity on ellipse lower half vs. a (E = 350 eV, b = 10)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("BestReflectivity_vs_a_350eV_Al_SiO2_b=10.png", dpi=300)
plt.show()

# --- Plot: percent of lower half with R >= threshold (additional) ---
plt.figure(figsize=(8,5))
for mat, values in percent_high.items():
    plt.plot(a_values, values, marker='o', label=f"{mat} (R ≥ {threshold})")
plt.xlabel("a (ellipse semi-major axis)")
plt.ylabel(f"Percentage of lower half with R ≥ {threshold}")
plt.title(f"Arc percentage with R ≥ {threshold} vs a (E = 350 eV, b = 10)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("PercentHighReflectivity_vs_a_350eV_Al_SiO2_b=10.png", dpi=300)
plt.show()

# --- NEW: Plot angles between the two rays (max-x and min-x good points) vs a ---
plt.figure(figsize=(8,5))
for mat, values in angles_between.items():
    # convert nan to np.nan for plotting; matplotlib will skip nan
    vals = np.array(values, dtype=float)
    plt.plot(a_values, vals, marker='^', linestyle='-', label=f"{mat} (angle between rays)")
plt.xlabel("a (ellipse semi-major axis)")
plt.ylabel("Angle between rays (degrees)")
plt.title("Angle between rays from F1 to min-x and max-x R≥0.8 points vs a")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("AnglesBetweenRays_vs_a_350eV_Al_SiO2_b=10.png", dpi=300)
plt.show()
