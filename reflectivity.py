"""
Author: Blerta Rushiti
Reflectivity Analysis for Elliptical Mirrors at 350 eV (Soft X-rays)

This script analyzes the reflectivity properties of an elliptical mirror by:
1. Calculating incident angles along the lower half of an ellipse (y ≤ 0)
2. Identifying regions with optimal reflectivity (incident angles > threshold)
3. Plotting the incident angle distribution along the x-axis
4. Loading and interpolating Henke data for various materials at 350 eV 

Parameters:
    a, b: semi-major and semi-minor axes of the ellipse
    seuil_angle: threshold angle (in degrees) for identifying high-reflectivity regions

The script generates plots showing:
- Incident angle distribution vs x position
- Ellipse geometry with highlighted high-reflectivity regions
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Ellipse parameters ---
a = 25
b = 1

# Foci
c = np.sqrt(a**2 - b**2)
F1 = (-c, 0)
F2 = (c, 0)

# Points on the ellipse
theta = np.linspace(0, 2*np.pi, 500)
x = a * np.cos(theta)
y = b * np.sin(theta)

# --- Incident angle calculation (interior normal) ---
angles_incident = []
x_lower = []
y_lower = []  # Ajoute ce tableau

for xi, yi in zip(x, y):
    if yi < -1e-12:  # only southern hemisphere
        if abs(yi) < 1e-6:  # avoid division by zero near y=0
            continue

        vi = np.array([xi - F1[0], yi - F1[1]])  # incident vector
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

        x_lower.append(xi)
        y_lower.append(yi)  # Ajoute yi ici
        angles_incident.append(theta_incident)

angles_incident = np.array(angles_incident)
x_lower = np.array(x_lower)
y_lower = np.array(y_lower)  # Convertis en array

# --- Trouver les points avec meilleure réflectivité (angle incident élevé) ---
seuil_angle = 70  # seuil en degrés, à ajuster selon ton besoin
mask_best = np.degrees(angles_incident) > seuil_angle

x_best = x_lower[mask_best]
y_best = y_lower[mask_best]  # Utilise y_lower ici
angles_best = np.degrees(angles_incident[mask_best])

print("Positions (x, y) avec meilleure réflectivité (angle incident > {:.1f}°):".format(seuil_angle))
for xi, yi, ai in zip(x_best, y_best, angles_best):
    print(f"x = {xi:.2f}, y = {yi:.2f}, angle incident = {ai:.2f}°")

# --- Plot incident angle ---
plt.figure(figsize=(8,5))
plt.plot(x_lower, np.degrees(angles_incident), 'b-', label="Incident angle (°)")
plt.xlabel("x position")
plt.ylabel("Incident angle (°)")
plt.title("Incident angle on the ellipse (y ≤ 0)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("IncidentAngle_350eV.png", dpi=300)
plt.show()

# --- Materials and Henke files ---
materials = {
    "Si": "Si_350eV.txt",
    "Al": "Al_350eV.txt",
    "Mo/Si": "MoSi_350eV.txt",
    "SiO2": "SiO2_350eV.txt",    
    "Ni": "Ni_350eV.txt",
    "Cr/Ti": "Cr-Ti_350eV.txt",
    "Cr/V": "Cr-V_350eV.txt",
    "Cr/Sc": "Cr-Sc_350eV.txt",
    "W/B4C": "W-B4C_350eV.txt"

}

# --- Convert incident angle -> grazing angle ---
angles_inc_deg = np.degrees(angles_incident)
theta_glancing_deg = 90 - angles_inc_deg
theta_glancing_deg = np.clip(theta_glancing_deg, 0, 90)

# --- Plot reflectivity (only for incident angles > 70°) ---
plt.figure(figsize=(8,5))
for mat, filename in materials.items():
    data = np.genfromtxt(filename, comments="#", usecols=(0,1))
    angles_henke = data[:,0]
    R_henke = data[:,1]

    interp_R = interp1d(angles_henke, R_henke, kind='linear',
                        bounds_error=False, fill_value=(R_henke[0], R_henke[-1]))
    
    R_points = interp_R(theta_glancing_deg)
    R_points = np.clip(R_points, 0, 1)

    # --- Filtrage pour n'afficher que les angles incidents > 70° ---
    mask = angles_inc_deg > 70
    plt.plot(angles_inc_deg[mask], R_points[mask], label=f"{mat}")

plt.xlabel("Incident angle (°) w.r.t. normal")
plt.ylabel("Reflectivity")
plt.title("Material reflectivity (E = 350 eV, angles > 70°)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("MaterialReflectivity_350eV_gt70.png", dpi=300)
plt.show()

# --- Find highest incident angle ---
max_idx = np.argmax(angles_incident)
theta_max = np.degrees(angles_incident[max_idx])
x_max = x_lower[max_idx]
y_max = y[y<0][max_idx]
theta_glancing_max = 90 - theta_max
theta_glancing_max = np.clip(theta_glancing_max, 0, 90)

thata_2 = 5
theta_2 = np.clip(thata_2, 0, 90)

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

    R_2 = float(interp_R(theta_2))
    R_2 = np.clip(R_2, 0, 1)
    
    print(f"{mat}: Reflectivity = {R_max:.4f}")
    print(f"  Reflectivity at {thata_2}° grazing angle: {R_2:.4f}")





# Optionnel : afficher ces points sur le graphe
plt.figure(figsize=(8,5))
plt.plot(x_lower, np.degrees(angles_incident), 'b-', label="Incident angle (°)")
plt.scatter(x_best, angles_best, color='red', label="Best reflectivity zone")
plt.xlabel("x position")
plt.ylabel("Incident angle (°)")
plt.title("Incident angle on the ellipse (y ≤ 0)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --- Tracer la partie de l'ellipse avec meilleure réflectivité et les rayons incidents ---
plt.figure(figsize=(12,4))

# Tracer toute l'ellipse en gris clair
plt.plot(x, y, color='lightgray', linewidth=1, label="Entier Ellipse")

# Tracer seulement la partie avec meilleure réflectivité en rouge
plt.plot(x_best, y_best, 'r-', linewidth=3, label="Best reflectivity")

# Tracer le foyer source
plt.scatter([F1[0]], [F1[1]], color='blue', marker='o', label="Focus F1")

# Tracer quelques rayons incidents (par exemple, 5 rayons répartis sur la zone rouge)
if len(x_best) > 0:
    idx_rays = np.linspace(0, len(x_best)-1, 5, dtype=int)
    for i in idx_rays:
        plt.plot([F1[0], x_best[i]], [F1[1], y_best[i]], 'g--', alpha=0.5)

    # --- AJOUT : tracer les deux droites spéciales (min-x et max-x parmi les points R≥seuil) ---
    i_max = np.argmax(x_best)
    i_min = np.argmin(x_best)
    p_max = (x_best[i_max], y_best[i_max])
    p_min = (x_best[i_min], y_best[i_min])

    # droites épaisses et couleurs distinctes
    plt.plot([F1[0], p_max[0]], [F1[1], p_max[1]], color='green', linewidth=2.5, label='Ray towards max-x (R≥thr)')
    plt.plot([F1[0], p_min[0]], [F1[1], p_min[1]], color='orange', linewidth=2.5, label='Ray towards min-x (R≥thr)')

    # marquer les deux points spéciaux
    plt.scatter([p_max[0], p_min[0]], [p_max[1], p_min[1]], color='magenta', s=80, edgecolor='k', zorder=5, label='Points min/max-x (R≥thr)')

    # calculer et afficher l'angle entre ces deux droites
    v1 = np.array([p_max[0] - F1[0], p_max[1] - F1[1]])
    v2 = np.array([p_min[0] - F1[0], p_min[1] - F1[1]])
    cosang = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cosang)))
    plt.text(0.02, 0.95, f"Angle between the to vectors = {angle_deg:.2f}°",
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

plt.xlabel("x")
plt.ylabel("y")
plt.title("Ellipse with Best Reflectivity Zone and Incident Rays")

# Déplacer la légende en dehors du graphe (à droite) pour ne pas masquer la figure
plt.gca().set_aspect('equal')
legend = plt.legend(loc='center left', bbox_to_anchor=(0.90, -1), fontsize='small')
plt.gca().add_artist(legend)

# Ajuster tight_layout pour laisser de la place à la légende
plt.tight_layout(rect=[0, 0, 0.85, 1.0])
plt.show()