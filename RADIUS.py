"""
Author: Blerta Rushiti
Conic Surface Visualization Tool

This script provides functionality to visualize conic surfaces commonly used in optics,
based on their radius of curvature (R) and conic constant (k).

Main features:
1. Draw conic sections including:
   - Ellipses (k > -1)
   - Parabolas (k = -1)
   - Hyperbolas (k < -1)

2. Implementation using the Zemax formula:
   z = (r²)/(R * (1 + sqrt(1 - (1+k)(r²/R²))))
Parameters:
    R: Radius of curvature at the vertex
    k: Conic constant
    n: Number of points to generate along the conic curve (default is 500)

Examples:
    draw_conic(R=125, k=24)    # Oblate ellipsoid
    draw_conic(R=1.8, k=-0.64) # Prolate ellipsoid

"""

import numpy as np
import matplotlib.pyplot as plt

def draw_conic(R, k, n=500):
    """
    Trace une conique (ellipse si k > -1) à partir de R et k.
    """
    # Domaine valide pour r
    if k > -1:  # ellipse
        r_max = R / np.sqrt(1 + k)
    else:
        r_max = R * 2   # valeur arbitraire pour parabole/hyperbole

    r = np.linspace(-r_max, r_max, n)

    # Formule de la conique (Zemax)
    inside = 1 - (1+k) * (r**2 / R**2)
    z = np.zeros_like(r)
    valid = inside >= 0
    z[valid] = (r[valid]**2) / (R * (1 + np.sqrt(inside[valid])))

    # Tracé
    plt.figure(figsize=(6,6))
    plt.plot(r, z, 'b')
    plt.gca().set_aspect("equal")
    plt.xlabel("r (mm)")
    plt.ylabel("z (mm)")
    plt.title(f"Conique (R={R}, k={k})")
    plt.grid(True)
    plt.show()

# Exemple
draw_conic(R=125, k=24)
draw_conic(R=1.8, k=-0.64)
