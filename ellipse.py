"""
Ellipse Incident Angle Calculator

This script calculates and visualizes incident angles on an elliptical mirror surface.
It demonstrates the geometric properties of elliptical mirrors by:
1. Defining an ellipse with semi-major axis a and semi-minor axis b
2. Calculating incident angles at specific points using the interior normal method
3. Visualizing the mirror surface, foci, and chosen points

Features:
- Calculates foci positions based on semi-axes
- Implements incident angle calculation using interior normal method
- Plots the lower half of the ellipse (mirror surface)
- Visualizes chosen points and their incident rays

Parameters:
    a: semi-major axis (= 25)
    b: semi-minor axis (= 1)
    theta0, theta1: angular positions of chosen points (-π/2, -π/4)

Functions:
    incident_angle_interior(F, point, a, b):
        Calculates the incident angle at a given point using the interior normal method

Output:
    - Console output of incident angles at chosen points
    - Plot showing mirror surface, foci, and chosen points

Author: Blerta Rushiti
"""

import numpy as np
import matplotlib.pyplot as plt

a = 25  # semi-major axis
b = 1   # semi-minor axis

# Foci
c = np.sqrt(a**2 - b**2)
print ("Le foyer est c =", c)
F1 = (-c, 0)
F2 = (c, 0)

theta = np.linspace(0, 2*np.pi, 300)
x = a * np.cos(theta)
y = b * np.sin(theta)

# Chosen points
theta0 = -np.pi/2
x0, y0 = a*np.cos(theta0), b*np.sin(theta0)

theta1 = -np.pi/4
x1, y1 = a*np.cos(theta1), b*np.sin(theta1)

# --- Function to compute incident angle using interior normal ---
def incident_angle_interior(F, point, a, b):
    xi, yi = point
    # Incident vector
    vi = np.array([xi - F[0], yi - F[1]])
    # Tangent and normal
    slope_tangent = -(b**2 * xi) / (a**2 * yi)
    tangent = np.array([1, slope_tangent])
    normal = np.array([-tangent[1], tangent[0]])  # perpendicular vector
    # Orient normal inward (towards center)
    if np.dot(normal, np.array([xi, yi])) > 0:
        normal = -normal
    # Angle between incident vector and normal
    uvi = vi / np.linalg.norm(vi)
    un = normal / np.linalg.norm(normal)
    cos_theta = np.clip(np.dot(uvi, un), -1, 1)
    theta_inc = np.arccos(cos_theta)
    # Ensure angle ≤ 90°
    if theta_inc < np.pi/2:
        theta_inc = np.pi - theta_inc
    return theta_inc, normal, vi

# --- Compute for both points ---
theta_inc0, normal0, vi0 = incident_angle_interior(F1, (x0, y0), a, b)
theta_inc1, normal1, vi1 = incident_angle_interior(F1, (x1, y1), a, b)

print("Point 0: Incident angle (deg):", np.degrees(theta_inc0))
print("Point 1: Incident angle (deg):", np.degrees(theta_inc1))

# --- Plot ---
plt.figure(figsize=(12,3))

# ellipse seulement au-dessus de l’axe x
mask = y < 0
plt.plot(x[mask], y[mask], label="Mirror surface")

plt.scatter([F1[0], F2[0]], [F1[1], F2[1]], color="red", marker="o", label="Foci")
plt.scatter(x0, y0, color="blue", marker="s", label="Chosen point 0")
plt.scatter(x1, y1, color="brown", marker="s", label="Chosen point 1")

# Rays
plt.plot([F1[0], x0], [F1[1], y0], 'g--')
plt.plot([x0, F2[0]], [y0, F2[1]], 'm--')
plt.plot([F1[0], x1], [F1[1], y1], 'g--')
plt.plot([x1, F2[0]], [y1, F2[1]], 'm--')

# Inward normals
plt.plot([x0, x0+normal0[0]], [y0, y0+normal0[1]], 'k-')
plt.plot([x1, x1+normal1[0]], [y1, y1+normal1[1]], 'k-')

plt.gca().set_aspect("equal")

plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.legend(loc="upper left", bbox_to_anchor=(0.7, 0.2))
plt.title("Ellipsoidal mirror with incident rays (y > 0)")
plt.show()
