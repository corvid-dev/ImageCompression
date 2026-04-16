"Import Libraries"
# Linear Algebra
import numpy as np
from numpy import ndarray
# Image Processing
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_float
# Visualization
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass(slots=True)
class SVDForm:
    A: np.ndarray
    U:  np.ndarray
    S:  np.ndarray
    Vh: np.ndarray
    @classmethod
    def from_matrix(cls, A: np.ndarray, full_matrices: bool = False) -> "SVDForm":
        U, S, Vh = np.linalg.svd(A, full_matrices=full_matrices)
        return cls(A=A, U=U, S=S, Vh=Vh)

def matrix_normalization(path, print=False):
    """
    Load an image from path and convert it to a normalized grayscale matrix.
    If print=False, then do not display the image conversion to the user.
    """
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

def orthogonality_check(svd):
    """
    Check for orthogonality by comparing products to identity matrices of same size.
    For any orthonormal matrix Q, Q^TQ=I.
    For compact SVD, U is m x r and and Vh is r x n.
    Hence, UtU = I_r and VVt = I_r.
    """
    #1. Extract svd form
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    #U, S, Vh = np.linalg.svd(array) # full svd
    UtU = svd.U.T @ svd.U
    VtV = svd.Vh @ svd.Vh.T
    # U  -> Unitary Matrix, Orthonormal
    # S  -> Diagonal of Σ, contains σ_i
    # Vh -> V^H, Hermitian

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

    #4. Compute distance between each matrix and identity matrix
    # https://numpy.org/doc/2.1/reference/generated/numpy.linalg.norm.html
    print("||UtU - I_u||:", np.linalg.norm(UtU - I_u))
    print("||VVt - I_v||:", np.linalg.norm(VtV - I_v))
    print("U numerically orthogonal:", is_u_close)
    print("V numerically orthogonal:", is_v_close)

def visualize_svd(svd):
    """
    Visualize the SVD transformation on test vector s_bar.
    Each step is shown in a 2D orthonormal basis built via Gram-Schmidt from the
    input and output vectors. The input vector is aligned to the x-axis.
    Each panel uses a different local basis, so angles and lengths are not directly
    comparable across panels.
    """
    m,n = svd.A.shape # m = row, n = col
    U, S, Vh = svd.U, svd.S, svd.Vh
    r = len(S) # r = rank
    s_bar = np.ones(n, dtype=float) # test vector <1,1,...,1>
    Sigma = np.diag(S)
    
    v_1 = Vh @ s_bar
    v_2 = Sigma @ v_1
    v_3 = U @ v_2

    def normalize(v):
        # turn any vector into a unit vector
        norm = np.linalg.norm(v)
        if norm < 1e-10: # check if its a zero vector
            return np.zeros_like(v) # return a zero vector of the same shape to avoid divbyzero
        return v / norm
   
    def ON_basis(v_b, v_a):
        # perform Gram-Schmidt to build orthonormal 2D basis
        q1 = normalize(v_b)
        v_perp = v_a - (q1.T @ v_a) * q1
        # guard against collinear v_before, v_after
        if np.linalg.norm(v_perp) < 1e-10:  # if the perpendicular vector is 0, use standard basis.
            e = np.zeros_like(q1)
            e[0] = 1.0
            if len(q1) > 1 and abs(q1.T @ e) > 0.9:
                e[0] = 0.0
                e[1] = 1.0
            v_perp = e - (q1.T @ e) * q1
        q2 = normalize(v_perp)  # otherwise use gram schmidt
        return np.column_stack([q1, q2])

    def embed(v,dim):
        E = np.zeros(dim, dtype=float) # build a new zeros matrix of size dim
        E[:len(v)]=v # embed the vector in the first column
        return E

    def get_panel_coords(v_before, v_after, dim):
        v_b = embed(v_before, dim)
        v_a = embed(v_after, dim)
        # construct orthonormal basis
        basis = ON_basis(v_b, v_a)
        # project onto plane
        c_before = basis.T@v_b
        c_after = basis.T@v_a
        return c_before, c_after

    # Construct vectors within 2d coordinate system
    # Panel 1: Apply V^H (rotation): s -> V^H s
    p1_s, p1_v1 = get_panel_coords(s_bar, v_1, n)

    # Panel 2: Apply Σ (scaling): V^H s -> Σ V^H s
    p2_v1, p2_v2 = get_panel_coords(v_1, v_2, r)

    # Panel 3: Apply U (rotation): Σ V^H s -> U Σ V^H s
    p3_v2, p3_v3 = get_panel_coords(v_2, v_3, m)

    panels = [
        (p1_s, p1_v1, r"$\bar{s}$", r"$V^H \bar{s}$", r"Rotation by $V^H$"),
        (p2_v1, p2_v2, r"$V^H \bar{s}$", r"$\Sigma V^H \bar{s}$", r"Scaling by $\Sigma$"),
        (p3_v2, p3_v3, r"$\Sigma V^H \bar{s}$", r"$U \Sigma V^H \bar{s}$", r"Rotation by $U$")
    ]

    # build plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    def signed_angle_2d(v_b, v_a):
        # guard against 0 length vectors
        if np.linalg.norm(v_b) < 1e-10 or np.linalg.norm(v_a) < 1e-10:
            return 0.0
        # check the signed 2d angle using the formula
        cross = v_b[0] * v_a[1] - v_b[1] * v_a[0]
        dot = v_b[0] * v_a[0] + v_b[1] * v_a[1]
        return np.degrees(np.arctan2(cross, dot))
    
    def length_change_perc(v_b, v_a):
        length_before = np.linalg.norm(v_b)
        length_after = np.linalg.norm(v_a)
        if length_before < 1e-10:
            pct_change = 0.0
        else:
            pct_change = ((length_after - length_before) / length_before) * 100
        if abs(pct_change) < 1e-5:
            pct_change = 0.0
        return pct_change

    # draw the plots
    for ax, (v_b, v_a, label_b, label_a, title) in zip(axes, panels): # extract relevant information from each panel.
        angle = signed_angle_2d(v_b, v_a)   # calculate the angle change
        pct_change = length_change_perc(v_b,v_a)    # calculate the length change
        info = plt.Line2D([], [], linestyle='none',
                    label=f"Rescale {pct_change:.1f}%\nθ = {angle:.1f}°")
        # determine plot limits based on vector magnitude, with some padding
        limit = max(np.linalg.norm(v_b), np.linalg.norm(v_a), 1) * 1.2
        # draw vectors from (0,0) to (vector_x,vector_y)
        ax.quiver(0, 0, v_b[0], v_b[1], angles='xy', scale_units='xy',
                scale=1, color='red', label=f"Before: {label_b}", alpha=0.7)
        ax.quiver(0, 0, v_a[0], v_a[1], angles='xy', scale_units='xy',
                scale=1, color='blue', label=f"After: {label_a}", alpha=0.4)
        # format axis
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(info)
        labels.append(info.get_label())
        ax.legend(handles, labels, fontsize='small')
        ax.set_title(title, fontsize=12)

    # display plots to user.
    plt.tight_layout()
    plt.show()

# Helper Function
def part4_5_grapher(title, array: ndarray, U: ndarray, S: ndarray, Vh: ndarray, LOG_SCALE: tuple):
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
        rel_error = np.linalg.norm(array - A_k, 'fro') / A_frobenius
        # Calculate the energy using:
        # Energy(k) = (∑^k_{i=1} σ^2_i) / (∑^n_{i=1} σ^2_i)
        energy_k = np.sum(S[:k] ** 2) / energy_denominator_sum

        # Add relative error and energy to the top of each plot
        axes[i].set_title(
            f"k = {k}\nError: {rel_error:.4f}\nEnergy: {energy_k:.4f}"
        )

        # Print metrics to console for not fancy viewing
        print(f"k = {k:>4} | Relative Error: {rel_error:.6f} | Energy: {energy_k:.6f}")

        # Plot A_k to one of the sub-graphs from above
        axes[i].imshow(A_k, cmap='gray')
        axes[i].axis('off')

    # Plot making and construction
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def spectral_analysis_and_error_quantification(svd):
    # Re-make the matrices U, S, Vh & Σ_i
    array, U, S, Vh = svd.A, svd.U, svd.S, svd.Vh

    # Tuple to iterate through the different LOG scales
    LOG_SCALE = (1,10,50,100) # can remove 71 for final submission

    # Plot singular values on a log scale
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.semilogy(S, color='steelblue')
    # Title/Axis Labels
    ax.set_title("Singular Values (Log Scale)")
    ax.set_xlabel("Index i")
    ax.set_ylabel(r"$\sigma_i$ (log scale)")

    # Other Configurations before making the graph
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    part4_5_grapher(r"Rank-k Approximations $(A_k)$", array, U, S, Vh, LOG_SCALE)

def compression_ratio(svd):
    # Re-make U, S & Vh again part 3
    array, U, S, Vh = svd.A, svd.U, svd.S, svd.Vh
    # ε = 0.001
    epsilon = 1e-3
    # m = num rows  |  n = num cols
    m, n = array.shape

    # ||A - A_k||_2 = σ_{k+1}, so find smallest k where σ_{k+1} < ε
    indices = np.where(S < epsilon * S[0])[0]
    k = indices[0] if len(indices) > 0 else len(S)

    # Calculate the compression ratio (CR) using:
    # CR = (m × n) / (k(m + n + 1))
    CR = (m*n)/(k*(m+n+1))

    # Prints out the optimal k value and the CR
    print(f"Optimal k: {k}")
    print(f"Compression Ratio: {CR:.6f}")

    original_img_k_val = min(m, n)
    part4_5_grapher(f"Optimal k-Value vs. Original Photo", array, U, S, Vh, (k, original_img_k_val))

file_path = "balloons.jpg"
image_array = matrix_normalization(file_path, print=True)
compactSVD = SVDForm.from_matrix(image_array, full_matrices=False) # if full_matrices=True, will use full SVD form.
orthogonality_check(compactSVD)
visualize_svd(compactSVD)
spectral_analysis_and_error_quantification(compactSVD)
compression_ratio(compactSVD)