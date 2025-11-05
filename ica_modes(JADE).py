import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# -------------------------------
# JADE ICA function (Python 3 compatible)
# -------------------------------
def jadeR(X, m=None):
    """
    JADE ICA
    X: data matrix (features x samples), should be whitened
    m: number of components to extract
    Returns:
        W: unmixing matrix (m x features)
        S: independent components (m x samples)
    """
    X = np.asarray(X)
    n, T = X.shape

    if m is None:
        m = n
    if m > n or m > T:
        raise ValueError("Number of components must be <= min(num_features, num_samples)")

    X = X[:m, :]  # select first m features if necessary

    # Compute approximate cumulant matrices (simplified for illustration)
    Q = []
    for i in range(m):
        Xi = X[i, :]
        for j in range(i, m):
            Xj = X[j, :]
            Qij = (Xi * Xj) @ X.T / T - np.diag(np.var(X, axis=1))
            Q.append(Qij)

    # Joint diagonalization via eigen decomposition (simplified)
    C = sum(Q) / len(Q)
    d, W = np.linalg.eigh(C)
    W = W.T
    S = W @ X
    return W, S

# -------------------------------
# Load images as columns
# -------------------------------
def load_images_as_columns(folder, target_size=(64,64)):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
            img = Image.open(os.path.join(folder, filename)).convert('L')
            img = img.resize(target_size)
            img_vec = np.array(img).flatten()
            images.append(img_vec)
    if len(images) == 0:
        raise ValueError("No images found in the folder.")
    X = np.column_stack(images)  # pixels x images
    return X

# -------------------------------
# Whitening
# -------------------------------
def whiten(X):
    X_mean = X.mean(axis=1, keepdims=True)
    X_centered = X - X_mean
    cov = np.cov(X_centered)
    d, E = np.linalg.eigh(cov)
    epsilon = 1e-6
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + epsilon))
    V = D_inv_sqrt @ E.T
    X_white = V @ X_centered
    return X_white, V, X_mean

# -------------------------------
# HSIC computation
# -------------------------------
def rbf_kernel(X, sigma=None):
    pairwise_sq_dists = np.sum((X[:, None] - X[None, :])**2, axis=2)
    if sigma is None:
        sigma = np.median(pairwise_sq_dists)
    K = np.exp(-pairwise_sq_dists / (2*sigma**2))
    return K

def hsic(X, Y):
    n = X.shape[0]
    H = np.eye(n) - np.ones((n,n))/n
    K = rbf_kernel(X.reshape(-1,1))
    L = rbf_kernel(Y.reshape(-1,1))
    return np.trace(K @ H @ L @ H) / ((n-1)**2)

def compute_hsic_matrix(S):
    n_modes = S.shape[0]
    hsic_mat = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(i, n_modes):
            val = hsic(S[i], S[j])
            hsic_mat[i,j] = val
            hsic_mat[j,i] = val
    return hsic_mat

# -------------------------------
# Full pipeline
# -------------------------------
folder = "rescaled_images"  # Folder with images
X = load_images_as_columns(folder, target_size=(64,64))  # pixels x images
pixels, n_samples = X.shape

# Whitening
X_white, V, X_mean = whiten(X)

# PCA reduction: n_components < n_samples and < pixels
n_components_pca = min(n_samples-1, pixels, 50)
pca = PCA(n_components=n_components_pca)
X_reduced = pca.fit_transform(X_white.T).T  # reduced_dims x samples

# JADE ICA
W, S = jadeR(X_reduced, m=n_components_pca)

# HSIC matrix computation
hsic_matrix = compute_hsic_matrix(S)
print("HSIC matrix:\n", hsic_matrix)

# -------------------------------
# Reconstruct ICA modes (basis vectors) in original pixel space
# -------------------------------
# Project ICA basis vectors W back to pixel space using PCA inverse transform
ica_basis_in_pixel_space = pca.inverse_transform(W)  # n_components x pixels
ica_basis_in_pixel_space += X_mean.ravel()  # add mean back per pixel

# -------------------------------
# Save all ICA modes as images
# -------------------------------
out_folder = "JADE_modes"
os.makedirs(out_folder, exist_ok=True)

for i in range(ica_basis_in_pixel_space.shape[0]):
    if ica_basis_in_pixel_space.shape[1] != 4096:
        print(f"Mode {i+1} wrong pixel count: {ica_basis_in_pixel_space.shape[1]}")
        continue
    mode_img = ica_basis_in_pixel_space[i].reshape(64, 64)
    # Normalize to uint8 for saving
    norm_img = (mode_img - np.min(mode_img)) / (np.ptp(mode_img) + 1e-7) * 255
    img_uint8 = Image.fromarray(norm_img.astype(np.uint8))
    img_uint8.save(os.path.join(out_folder, f"JADE_mode_{i+1:03d}.png"))

print(f"All modes saved to folder: {out_folder}")

# -------------------------------
# HSIC heatmap for first 20 ICA modes (color only, no numbers)
# -------------------------------
num_plot = min(20, hsic_matrix.shape[0])
hsic_20 = hsic_matrix[:num_plot, :num_plot]

plt.figure(figsize=(8, 7))
sns.heatmap(hsic_20, annot=False, cmap="viridis")
plt.title("HSIC Heatmap (First 20 ICA modes)")
plt.xlabel("Modes")
plt.ylabel("Modes")
plt.tight_layout()
plt.show()

print("""
HSIC heatmap color interpretation:
- **Dark purple/blue** (low values): Indicates **higher independence** between modes.
- **Yellow/green/lighter colors** (high values): Indicates **stronger dependence** (less independence) between modes.

Ideally, for independent components, most off-diagonal elements should show dark (low) colors.
""")
