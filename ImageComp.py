"Import Libraries"
# Linear Algebra
import numpy as np
# Image Processing
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_float
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def matrix_normalization(path, print=False):
    # https://scikit-image.org/docs/0.24.x/api/skimage.html
    #1. Load the path and extract image.
    img = io.imread(path)

    #2. Normalize, convert image to float64 and scale between [0,1].
    img_float = img_as_float(img)

    #2. Make grayscale.
    # Check if already grayscale
    if img_float.ndim == 2: 
        A = img_float
    # Otherwise convert to grayscale
    else: 
        # Handle RGBA by removing alpha channel if it exists
        A = rgb2gray(img_float[:, :, :3])

    if(print):
        # 3. Display converted image.
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(A, cmap='gray')
        axes[1].set_title("Grayscale")
        axes[1].axis('off')
        plt.show()

    # 4. Return array of normalized floats
    return A

def orthogonality_check(array):
    #1. Extract svd form
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    #U, S, Vh = np.linalg.svd(array) # full svd
    U, S, Vh = np.linalg.svd(array, full_matrices = False) # compact svd
    # U  -> Unitary Matrix, Orthonormal
    # S  -> Diagonal of Σ, contains σ_i
    # Vh -> V^H, Hermitian
    UtU = U.T @ U
    VtV = Vh @ Vh.T

    #2. Make Identity Matrices for comparison
    # https://numpy.org/devdocs/reference/generated/numpy.eye.html
    I_u = np.eye(UtU.shape[0]) # Create I of size u for comparison
    I_v = np.eye(VtV.shape[0]) # Create I of size v for comparison
    
    #3. Check if close
    # https://numpy.org/devdocs/reference/generated/numpy.isclose.html
    # Strict Tolerance
    is_u_close = np.allclose(UtU, I_u)
    is_v_close = np.allclose(VtV, I_v)
    
    # Relaxed tolerance
    #is_u_close = np.allclose(UtU, I_u, atol=1e-5)
    #is_v_close = np.allclose(VtV, I_v, atol=1e-5)
    print("U orthogonal:", is_u_close)
    print("V orthogonal:", is_v_close)
    
    #4. Compute distance between each matrix and identity matrix
    # https://numpy.org/doc/2.1/reference/generated/numpy.linalg.norm.html
    print("||UtU - I_u||:", np.linalg.norm(UtU - I_u))
    print("||VtV - I_v||:", np.linalg.norm(VtV - I_v))

def visualize_svd(array):
    #1. Extract compact svd form
    U, S, Vh = np.linalg.svd(array, full_matrices=False) # compact SVD

    #2. Define arbitrary test vector s_bar in R^2
    s_bar = np.array([1.0, 0.5])

    #3. Take first 2 components for 2D visualization
    Sigma_2 = np.diag(S[:2]) # Stretch vector along axis, build from  σ_i
    U_2 = U[:2, :2]  # Perform final rotation to put vector into output space
    Vh_2 = Vh[:2, :2] # Rotate test vector s_bar to align with principal components

    #4. Build vectors from transformations
    v_1 = Vh_2 @ s_bar              # V^H s
    v_2 = Sigma_2 @ v_1              # Σ V^H s
    v_3 = U_2 @ v_2                  # U Σ V^H s

    #5. Plot vectors
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # build axis
    def draw(ax, vec, title):
        ax.quiver(
            0, 0,
            vec[0], vec[1],
            angles='xy',
            scale_units='xy',
            scale=1
        )

        # scale axes to this vector
        max_val = max(abs(vec[0]), abs(vec[1]))

        # avoid zero range
        if max_val == 0:
            max_val = 1

        pad = 0.3 * max_val

        ax.set_xlim(-max_val - pad, max_val + pad)
        ax.set_ylim(-max_val - pad, max_val + pad)

        ax.axhline(0)
        ax.axvline(0)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid()

    # draw vectors upon axis
    draw(axes[0], v_1, r"$V^H \bar{s}$")
    draw(axes[1], v_2, r"$\Sigma V^H \bar{s}$")
    draw(axes[2], v_3, r"$U\Sigma V^H \bar{s}$")

    plt.tight_layout()
    plt.show()

def visualize_svd_vibe(matrix):
    """
    Visualize the compact SVD of a matrix using true 2D planes for each stage.

    The decomposition is:
        A = U Σ V^H

    The three panels show:
        1. s vs V^H s
        2. V^H s vs Σ V^H s
        3. Σ V^H s vs U Σ V^H s

    Each panel is drawn in an orthonormal basis for the plane spanned by the
    two vectors being compared, so lengths/angles within a panel are represented
    faithfully.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix of shape (m, n).
    """
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")

    m, n = A.shape

    # Compact SVD: A = U Σ Vh
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    r = len(S)

    # Default test vector in the domain R^n
    s = np.ones(n, dtype=float)

    def unit(v, tol=1e-12):
        norm = np.linalg.norm(v)
        if norm < tol:
            raise ValueError("Cannot normalize a near-zero vector.")
        return v / norm

    def plane_basis(a, b, tol=1e-12):
        """
        Return an orthonormal basis (e1, e2) for span{a, b}.
        If a and b are collinear, choose any perpendicular direction.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        e1 = unit(a, tol=tol)

        b_perp = b - np.dot(b, e1) * e1
        b_perp_norm = np.linalg.norm(b_perp)

        if b_perp_norm < tol:
            for k in range(len(a)):
                candidate = np.zeros_like(a)
                candidate[k] = 1.0
                candidate -= np.dot(candidate, e1) * e1
                cand_norm = np.linalg.norm(candidate)
                if cand_norm >= tol:
                    e2 = candidate / cand_norm
                    return e1, e2
            raise ValueError("Failed to construct a second basis vector.")

        e2 = b_perp / b_perp_norm
        return e1, e2

    def coords_in_plane(v, basis):
        e1, e2 = basis
        return np.array([np.dot(v, e1), np.dot(v, e2)])

    def embed(vec, dim):
        out = np.zeros(dim, dtype=float)
        out[:len(vec)] = vec
        return out

    def draw_panel(ax, before, after, before_label, after_label, title):
        max_val = max(np.max(np.abs(before)), np.max(np.abs(after)), 1e-6)
        pad = 0.12 * max_val
        limit = max_val + pad

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")

        # Draw axes FIRST (low z-order)
        ax.axhline(0, color="black", linewidth=1, zorder=0)
        ax.axvline(0, color="black", linewidth=1, zorder=0)

        # Draw vectors LAST (high z-order)
        ax.quiver(
            0, 0, before[0], before[1],
            angles="xy", scale_units="xy", scale=1,
            color="red", alpha=0.8, label=before_label,
            zorder=3
        )
        ax.quiver(
            0, 0, after[0], after[1],
            angles="xy", scale_units="xy", scale=1,
            color="blue", alpha=0.5, label=after_label,
            zorder=4
        )

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize="small")

    # Step 1: V^H s
    v1 = Vh @ s                   # shape (r,)

    # Step 2: Σ V^H s
    Sigma = np.diag(S)
    v2 = Sigma @ v1               # shape (r,)

    # Step 3: U Σ V^H s
    v3 = U @ v2                   # shape (m,)

    # Panel 1: s and v1 need the same ambient space
    v1_embed_n = embed(v1, n)

    # Panel 3: v2 and v3 need the same ambient space
    v2_embed_m = embed(v2, m)

    # Build exact 2D coordinates for each comparison plane
    basis1 = plane_basis(s, v1_embed_n)
    s_2d = coords_in_plane(s, basis1)
    v1_2d = coords_in_plane(v1_embed_n, basis1)

    basis2 = plane_basis(v1, v2)
    v1b_2d = coords_in_plane(v1, basis2)
    v2_2d = coords_in_plane(v2, basis2)

    basis3 = plane_basis(v2_embed_m, v3)
    v2b_2d = coords_in_plane(v2_embed_m, basis3)
    v3_2d = coords_in_plane(v3, basis3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    draw_panel(
        axes[0],
        s_2d,
        v1_2d,
        r"$\bar{s}$",
        r"$V^H \bar{s}$",
        r"$V^H \bar{s}$"
    )
    draw_panel(
        axes[1],
        v1b_2d,
        v2_2d,
        r"$V^H \bar{s}$",
        r"$\Sigma V^H \bar{s}$",
        r"$\Sigma V^H \bar{s}$"
    )
    draw_panel(
        axes[2],
        v2b_2d,
        v3_2d,
        r"$\Sigma V^H \bar{s}$",
        r"$U \Sigma V^H \bar{s}$",
        r"$U \Sigma V^H \bar{s}$"
    )

    plt.tight_layout()
    plt.show()

def spectral_analysis_and_error_quantification(array):
    # Re-make the matrices U, S, Vh & Σ_i
    U, S, Vh = np.linalg.svd(array, full_matrices=False)  # compact svd
    Sigma_i = np.diag(S[:])
    # Tuple to iterate through the different LOG scales
    LOG_SCALE = (1,10,50,100, 71) # can remove 71 for final submission

    # Plot singular values on a log scale
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(Sigma_i, color='steelblue')
    # Title/Axis Labels
    ax.set_title("Singular Values (Log Scale)")
    ax.set_xlabel("Index i")
    ax.set_ylabel(r"$\sigma_i$ (log scale)")
    # Other Configurations before making the graph
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Makes the Frobenius norm of A so it isn't calculated every iteration
    #  ||A||_F = sqrt{ sum_{i=1}^m sum_{j=1}^n |a_{ij}|^2 }
    A_frobenius = np.linalg.norm(array, 'fro')
    # Calculates the constant denominator for Energy(k) to reduce # calculations
    energy_denominator_sum = np.sum(S ** 2)

    # Makes a number of subplots equal to the length of LOG_SCALE
    fig, axes = plt.subplots(1, len(LOG_SCALE), figsize=(16, 4))

    # Loops through the values in LOG_SCALE and assigns the values index to i & the value to k
    for i, k in enumerate(LOG_SCALE):
        # Calculates A_k using:
        # ∑^k_{i=1} (σ_i u_i v^T)
        A_k = (U[:, :k] * S[:k]) @ Vh[:k, :]
        # Calculate the relative error using:
        # Error = (||A − A_k||_F) / (||A||_F)
        rel_error = np.linalg.norm(array-A_k, 'fro') / A_frobenius
        # Calculate the energy using:
        # Energy(k) = (∑^k_{i=1} σ^2_i) / (∑^n_{i=1} σ^2_i)
        energy_k = np.sum(S[:k] ** 2)/energy_denominator_sum

        # Plot A_k to one of the sub-graphs from above
        axes[i].imshow(A_k, cmap='gray')
        axes[i].set_title(
            f"k = {k}\nError: {rel_error:.4f}\nEnergy: {energy_k:.4f}"
        )
        axes[i].axis('off')

        # Print metrics to console for not fancy viewing
        print(f"k = {k:>4} | Relative Error: {rel_error:.6f} | Energy: {energy_k:.6f}")

    # Plot making and construction
    plt.suptitle(r"Rank-k Approximations $(A_k)$", fontsize=14)
    plt.tight_layout()
    plt.show()

def compression_ratio(array):
    # Re-make U, S & Vh again part 3
    U, S, Vh = np.linalg.svd(array, full_matrices=False)  # compact svd
    # ε = 0.001
    epsilon = 1e-3
    # m = num rows  |  n = num cols
    m, n = array.shape

    # ----- This is where stuff from part 3 should come in with 2-norm stuff -----

    # ||A - A_k||_2 = σ_{k+1}, so find smallest k where σ_{k+1} < ε
    k = None
    for i in range(len(S) - 1):
        if S[i + 1] < epsilon:
            k = i + 1
            break
    # Edge case: all singular values satisfy tolerance
    if k is None:
        k = len(S)

    # Calculate the compression ratio (CR) using:
    # CR = (m × n) / (k(m + n + 1))
    CR = (m*n)/(k*(m+n+1))
    # **Prints out CR**
    print(f"Compression Ratio: {CR:.6f}")

    # Theory on a hunch:
    # The decimal given by this function, you multiply that by 100 to get the optimal compression rate per photo

    # TODO:
    # Possibly add a way to show this working on the photo input
    # Make TS look nice

file_path = "veggies.png"
image_array = matrix_normalization(file_path)
orthogonality_check(image_array)
#visualize_svd(image_array)
visualize_svd_vibe(image_array)
spectral_analysis_and_error_quantification(image_array)
compression_ratio(image_array)