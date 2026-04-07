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

def matrix_normalization(path):
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
            0, 0,                 # start at origin
            vec[0], vec[1],       # vector components
            angles='xy',
            scale_units='xy',
            scale=1
        )

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
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


file_path = "veggies.png"
image_array = matrix_normalization(file_path)
orthogonality_check(image_array)
visualize_svd(image_array)