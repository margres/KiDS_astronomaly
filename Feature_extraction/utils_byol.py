import numpy as np
from sklearn.decomposition import PCA
import cv2
from astropy.stats import sigma_clipped_stats
from sklearn.preprocessing import MinMaxScaler

def run_pca(features, variance_threshold=0.98, use_scaler=False):
    """
    Perform PCA on the given features to retain the specified amount of variance.

    Parameters:
    features (array-like): The input features to perform PCA on.
    variance_threshold (float): The proportion of variance to retain.

    Returns:
    pca_result (array-like): The transformed features after PCA.
    """
    # Convert features to numpy array if not already
    features = np.array(features)
    if use_scaler:
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features.copy())
    
    # Initialize PCA with enough components to capture the desired variance
    pca = PCA(n_components=variance_threshold)

    # Fit and transform the features
    pca_result = pca.fit_transform(features)

    # Get the percentage of variance explained by each of the selected components
    explained_variance_ratio = pca.explained_variance_ratio_
    n_components = pca.n_components_

    # Print the results
    print(f"Number of components to retain {variance_threshold * 100}% of the variance: {n_components}")

    return pca_result


def denoise_bilateral_img(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

def sigma_clipping_gray(img, sigma=2, central=True):
    """
    Applies sigma clipping to the grayscale version of an RGB image, fits contours, and masks out regions.
    The mask is then applied to the original RGB image.

    Parameters
    ----------
    img : np.ndarray
        Input RGB image
    sigma : float, optional
        Sigma value for sigma clipping
    central : bool, optional
        If True, only the contour that contains the central point of the image is used

    Returns
    -------
    np.ndarray
        Processed image with regions outside the detected contour masked out
    """
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.nan_to_num(gray_img)  # OpenCV can't handle NaNs

    # Apply sigma clipping to the grayscale image
    mean, median, std = sigma_clipped_stats(gray_img, sigma=sigma)
    thresh = std + median
    img_bin = np.zeros(gray_img.shape, dtype=np.uint8)

    img_bin[gray_img <= thresh] = 0
    img_bin[gray_img > thresh] = 1

    # Find contours in the binary image
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x0 = gray_img.shape[0] // 2
    y0 = gray_img.shape[1] // 2

    selected_contour = None

    if central:
        for c in contours:
            if cv2.pointPolygonTest(c, (x0, y0), False) == 1:
                selected_contour = c
                break

    contour_mask = np.zeros_like(gray_img, dtype=np.uint8)
    if selected_contour is None:
        # This happens if there's no data in the image so we just return zeros
        return np.zeros_like(img)

    cv2.drawContours(contour_mask, [selected_contour], 0, (1, 1, 1), -1)
    
    # Apply the mask to each channel of the original RGB image
    new_img = np.zeros_like(img)
    for i in range(3):
        new_img[:, :, i] = cv2.bitwise_and(img[:, :, i], img[:, :, i], mask=contour_mask)
   

    return new_img

def sigma_70pxsquare_clipping_gray(img, sigma=2, central=True):
    """
    Applies sigma clipping to the grayscale version of an RGB image, fits contours, and masks out regions.
    The mask is then applied to the original RGB image.

    Parameters
    ----------
    img : np.ndarray
        Input RGB image
    sigma : float, optional
        Sigma value for sigma clipping
    central : bool, optional
        If True, only the contour that contains the central point of the image is used

    Returns
    -------
    np.ndarray
        Processed image with regions outside the detected contour masked out
    """

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.nan_to_num(gray_img)  # OpenCV can't handle NaNs

    # Determine the center of the image
    x0 = gray_img.shape[0] // 2
    y0 = gray_img.shape[1] // 2

    # Extract a 50x50 square centered at the image center
    half_side = 35
    square = gray_img[x0 - half_side:x0 + half_side, y0 - half_side:y0 + half_side]

    # Apply sigma clipping to the 50x50 square
    mean, median, std = sigma_clipped_stats(square, sigma=sigma)
    thresh = std + median
    img_bin = np.zeros(gray_img.shape, dtype=np.uint8)

    img_bin[gray_img <= thresh] = 0
    img_bin[gray_img > thresh] = 1

    # Find contours in the binary image
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    selected_contour = None

    if central:
        for c in contours:
            if cv2.pointPolygonTest(c, (x0, y0), False) == 1:
                selected_contour = c
                break

    contour_mask = np.zeros_like(gray_img, dtype=np.uint8)
    if selected_contour is None:
        # This happens if there's no data in the image so we just return zeros
        return np.zeros_like(img)

    cv2.drawContours(contour_mask, [selected_contour], 0, (1, 1, 1), -1)

    # Apply the mask to each channel of the original RGB image
    new_img = np.zeros_like(img)
    for i in range(3):
        new_img[:, :, i] = cv2.bitwise_and(img[:, :, i], img[:, :, i], mask=contour_mask)

    return new_img
