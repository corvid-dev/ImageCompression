
from skimage import io, img_as_float
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def matrix_normalization(img):
    path = img
    img = io.imread(path)
    img = img_as_float(img)
    # Check array dimensions, if 2, then already white or black
    A = rgb2gray(img)
    return A.astype(np.float32)

def orthogonality_check(array):
    #1. Extract svd form
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    U, S, Vh = np.linalg.svd(array) # full svd
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
    #is_u_close = np.allclose(UtU, I_u)
    #is_v_close = np.allclose(VtV, I_v)
    
    # Relaxed tolerance
    is_u_close = np.allclose(UtU, I_u, atol=1e-5)
    is_v_close = np.allclose(VtV, I_v, atol=1e-5)
    print("U orthogonal:", is_u_close)
    print("V orthogonal:", is_v_close)
    
    #4. Compute distance between each matrix and identity matrix
    # https://numpy.org/doc/2.1/reference/generated/numpy.linalg.norm.html
    print("||UtU - I_u||:", np.linalg.norm(UtU - I_u))
    print("||VtV - I_v||:", np.linalg.norm(VtV - I_v))


file_path = "white.png"
image_array = matrix_normalization(file_path)
orthogonality_check(image_array)