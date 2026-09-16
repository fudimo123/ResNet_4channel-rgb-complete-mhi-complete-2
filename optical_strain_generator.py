import cv2
import numpy as np

def generate_optical_strain(flow):
    """
    Calculates the normal components (exx, eyy) of the optical strain from the optical flow.

    Args:
        flow (np.ndarray): The optical flow field, with shape (h, w, 2).

    Returns:
        np.ndarray: A 2-channel array representing the xx and yy components of the normal strain.
    """
    # Separate the flow into u and v components
    u = flow[..., 0]
    v = flow[..., 1]

    # Calculate the gradients using the Sobel operator
    # ksize=3 means a 3x3 Sobel kernel
    du_dx = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
    dv_dy = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3)

    # The normal strain components are exx = du/dx and eyy = dv/dy
    exx = du_dx
    eyy = dv_dy

    # Stack the components into a 2-channel image
    strain_field = np.stack([exx, eyy], axis=-1)

    return strain_field