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
    m,n = array.shape # m = row, n = col
    U,S,Vh = np.linalg.svd(array, full_matrices=False)
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

    # def on_basis(a,b):
    #     # perform gram schmidt to build orthonormal basis
    #     q_1 = normalize(a) # normalize v_1 and assign to q_1
    #     b_perp = b - q_1.T @ b * q_1 # find b_perp to q_1
    #     q_2 = normalize(b_perp) # normalize b_perp and assign to q_2
    #     basis = np.column_stack([q_1,q_2]) # build matrix from basis vectors
    #     return basis 

    def on_basis(a, b):
        # 1. Normalize a. If zero, default to x-axis [1, 0, 0...]
        q_1 = normalize(a)
        if np.all(q_1 == 0): 
            q_1[0] = 1.0

        # 2. Compute perpendicular component
        b_perp = b - q_1.T @ b * q_1
        q_2 = normalize(b_perp)

        # 3. If q_2 is zero (b was zero or parallel to a), find a new direction
        if np.all(q_2 == 0):
            # Pick the axis q_1 points at the LEAST to ensure independence
            rescue = np.zeros_like(q_1)
            rescue[np.argmin(np.abs(q_1))] = 1.0
            q_2 = normalize(rescue - q_1.T @ rescue * q_1)

        return np.column_stack([q_1, q_2])
    def embed(v,dim):
        E = np.zeros(dim, dtype=float) # build a new zeros matrix of size dim
        E[:len(v)]=v # embed the vector in the first column
        return E
    
    def coord_plane(v,basis):
        # build a new 2-dimensional coordinate plane
        return basis.T @ v
    
    def get_panel_coords(v_before, v_after, dim):
        v_a = embed(v_before, dim)
        v_b = embed(v_after, dim)
        # construct orthonormal basis
        basis = on_basis(v_a, v_b)
        # project onto plane
        c_before = coord_plane(v_a, basis)
        c_after = coord_plane(v_b, basis)
        return c_before, c_after

    # Construct vectors within 2d coordinate system
    # Panel 1: Apply V^H (rotation): s -> V^H s
    p1_s, p1_v1 = get_panel_coords(s_bar, v_1, n)

    # Panel 2: Apply Σ (scaling): V^H s -> Σ V^H s
    p2_v1, p2_v2 = get_panel_coords(v_1, v_2, r)

    # Panel 3: Apply U (rotation): Σ V^H s -> U Σ V^H s
    p3_v2, p3_v3 = get_panel_coords(v_2, v_3, m)

    panels = [
        (p1_s, p1_v1, r"$\bar{s}$", r"$V^H \bar{s}$", "Rotation by $V^H$"),
        (p2_v1, p2_v2, r"$V^H \bar{s}$", r"$\Sigma V^H \bar{s}$", "Scaling by $\Sigma$"),
        (p3_v2, p3_v3, r"$\Sigma V^H \bar{s}$", r"$U \Sigma V^H \bar{s}$", "Rotation by $U$")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (before, after, label_a, label_b, title) in zip(axes, panels):
        # determine plot limits based on vector magnitude, with some padding
        limit = max(np.linalg.norm(before), np.linalg.norm(after), 1) * 1.2
        # draw vectors
        ax.quiver(0, 0, before[0], before[1], angles='xy', scale_units='xy', 
                scale=1, color='red', label=f"Before: {label_a}", alpha=0.7)
        ax.quiver(0, 0, after[0], after[1], angles='xy', scale_units='xy', 
                scale=1, color='blue', label=f"After: {label_b}", alpha=0.4)
        # format axis
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize='small')
        # print titles
        ax.set_title(title, fontsize=12)

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

file_path = "towson.jpg"
image_array = matrix_normalization(file_path, True)
orthogonality_check(image_array)
visualize_svd(image_array)
spectral_analysis_and_error_quantification(image_array)
compression_ratio(image_array)