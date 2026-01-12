import kaleido
import numpy as np
import plotly.graph_objects as go
import imageio.v2 as imageio
import os
import plotly.io as pio # Explicitly import plotly.io

# ------------------------------------------------------------
# Loss C(w) and gradient ∇C(w) for linear regression with 2 params
# y_hat = X @ w,   w = [w1, w2]
# C(w) = (1/(2N)) ||Xw - y||^2
# ∇C(w) = (1/N) X^T (Xw - y)
# ------------------------------------------------------------
X = np.array([
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 1.0],
    [3.0, 1.0],
    [4.0, 1.0],
])
y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
N = X.shape[0]

def loss(w):
    r = X @ w - y
    return 0.5 / N * (r @ r)

def grad(w):
    r = X @ w - y
    return (1.0 / N) * (X.T @ r)

# ------------------------------------------------------------
# Create grid for 3D surface
# ------------------------------------------------------------
w1_vals = np.linspace(-1.0, 4.0, 90)
w2_vals = np.linspace(-1.0, 4.0, 90)
W1, W2 = np.meshgrid(w1_vals, w2_vals)

C = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        C[i, j] = loss(np.array([W1[i, j], W2[i, j]]))

# ------------------------------------------------------------
# Gradient descent path
# ------------------------------------------------------------
w = np.array([-0.5, 3.5])   # start
eta = 0.2
steps = 25

path = [w.copy()]
for _ in range(steps):
    w = w - eta * grad(w)
    path.append(w.copy())
path = np.array(path)

path_z = np.array([loss(p) for p in path])

# ------------------------------------------------------------
# (Optional) Gradient vectors as cones (use -∇C for "downhill")
# Plot at a small constant z-offset to keep it readable
# ------------------------------------------------------------
show_cones = True

cones = None
if show_cones:
    w1_q = np.linspace(-1.0, 4.0, 12)
    w2_q = np.linspace(-1.0, 4.0, 12)
    W1q, W2q = np.meshgrid(w1_q, w2_q)

    Xc = W1q.flatten()
    Yc = W2q.flatten()

    # Place cones slightly above the minimum visible z for clarity
    z_offset = float(np.min(C)) + 0.2
    Zc = np.full_like(Xc, z_offset, dtype=float)

    U = np.zeros_like(Xc, dtype=float)
    V = np.zeros_like(Yc, dtype=float)
    W = np.zeros_like(Zc, dtype=float)

    for k in range(len(Xc)):
        g = grad(np.array([Xc[k], Yc[k]]))
        # negative gradient points downhill
        U[k] = -g[0]
        V[k] = -g[1]
        W[k] = 0.0

    cones = go.Cone(
        x=Xc, y=Yc, z=Zc,
        u=U, v=V, w=W,
        sizemode="scaled",
        sizeref=0.6,
        anchor="tail",
        name="-∇C (downhill direction)",
        showscale=False,
        opacity=0.7
    )

# ------------------------------------------------------------
# Plotly 3D figure
# ------------------------------------------------------------
surface = go.Surface(
    x=W1, y=W2, z=C,
    name="Loss surface C(w1,w2)",
    contours=dict(
        z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
    ),
    opacity=0.95
)

path_trace = go.Scatter3d(
    x=path[:, 0], y=path[:, 1], z=path_z,
    mode="lines+markers",
    name="Gradient descent path",
    marker=dict(size=4),
    line=dict(width=6)
)

opt_trace = go.Scatter3d(
    x=[2.0], y=[1.0], z=[loss(np.array([2.0, 1.0]))],
    mode="markers",
    name="Near optimum (2,1)",
    marker=dict(size=7, symbol="x")
)

data = [surface, path_trace, opt_trace]
if cones is not None:
    data.append(cones)

fig = go.Figure(data=data)

fig.update_layout(
    title="3D Loss Surface with Gradient Descent Path (and optional -∇C cones)",
    scene=dict(
        xaxis_title="w1",
        yaxis_title="w2",
        zaxis_title="C(w1,w2)",
    ),
    width=950,
    height=700,
)

fig.show()

# Ensure the figure is created before trying to save it
# fig variable is already available from the previous cell.

# Define the output directory for frames
output_dir = 'plot_frames'
os.makedirs(output_dir, exist_ok=True)

# List to store the paths of the saved frames
filepaths = []

# Define camera rotation parameters
num_frames = 60  # Number of frames for the GIF

# Initial camera position
initial_eye = dict(x=1.25, y=1.25, z=1.25)
fig.update_layout(scene_camera_eye=initial_eye)

for i in range(num_frames):
    # Calculate new camera position for rotation around the z-axis
    # This rotates the view around the center of the plot
    angle = 2 * np.pi * i / num_frames
    radius = 2.0  # Distance from the center of the plot
    new_x = radius * np.cos(angle)
    new_y = radius * np.sin(angle)

    # Update camera eye position, keep z constant for horizontal rotation
    fig.update_layout(scene_camera_eye=dict(x=new_x, y=new_y, z=1.25))

    # Save frame
    frame_filename = os.path.join(output_dir, f'frame_{i:03d}.png')
    pio.write_image(fig, frame_filename) # Use pio.write_image directly
    filepaths.append(frame_filename)

# Create GIF
gif_filename = 'loss_surface_gradient_descent.gif'
with imageio.get_writer(gif_filename, mode='I', fps=10) as writer:
    for filename in filepaths:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"GIF saved as {gif_filename}")

# Clean up individual frames
for filename in filepaths:
    os.remove(filename)
os.rmdir(output_dir)