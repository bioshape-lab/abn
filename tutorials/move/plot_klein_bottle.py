import matplotlib.pyplot as plt
import numpy as np


def klein_sphere_points(num_points, radius=1):
    """
    Generate points on a Klein sphere.

    Parameters:
    - num_points: Number of points to generate.

    Returns:
    - points: Array of points on the Klein sphere.
    """
    rng = np.random.default_rng(seed=0)
    # Generate random points on a 2D plane
    theta = rng.uniform(0, 2 * np.pi, num_points)
    phi = rng.uniform(0, 2 * np.pi, num_points)

    # Parametric equations for a Klein sphere
    x = radius * (np.cos(theta) * (1 + np.sin(phi)))
    y = radius * (np.sin(theta) * (1 + np.sin(phi)))
    z = radius * (np.sin(phi) * np.cos(phi))

    return np.column_stack((x, y, z))


def plot_klein_sphere(points):
    """
    Plot points on the Klein sphere.

    Parameters:
    - points: Array of points to plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="r", marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("Klein Sphere")

    plt.show()


# Generate points on the Klein sphere
num_points = 1000
scales = [0.5, 1, 0.5]
points = klein_sphere_points(num_points, scales)

# Plot the generated points
plot_klein_sphere(points)

# https://mathworld.wolfram.com/KleinBottle.html
# Klein Bagel


def klein_bottle_points(num_points, scale=1):
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, 2 * np.pi, num_points)
    U, V = np.meshgrid(u, v)

    X = (scale + np.cos(U / 2) * np.sin(V) - np.sin(U / 2) * np.sin(2 * V)) * np.cos(U)
    Y = (scale + np.cos(U / 2) * np.sin(V) - np.sin(U / 2) * np.sin(2 * V)) * np.sin(U)
    Z = np.sin(U / 2) * np.sin(V) + np.cos(U / 2) * np.sin(2 * V)
    return X, Y, Z


def plot_klein_bottle(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Klein Bottle")
    plt.show()


# Example usage
num_points = 100
scale = 3
X, Y, Z = klein_bottle_points(num_points, scale)
plot_klein_bottle(X, Y, Z)
