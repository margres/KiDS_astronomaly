import numpy as np
from skimage.transform import resize
import cv2
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
import cv2


def image_transform_log(img):
    """
    Normalise and then perform log transform on image

    Parameters
    ----------
    img : np.ndarray
        Input image (assumed float values)

    Returns
    -------
    np.ndarray
        Transformed image

    """

    mini = img[img != 0].min()
    maxi = img.max()
    offset = (maxi - mini) / 100

    if maxi == 0 and mini == 0:
        img = img + 0.01
    else:
        img = (img - mini) / (maxi - mini) + offset

    return np.log(img)


def image_transform_inverse_sinh(img):
    """
    Performs inverse hyperbolic sine transform on image

    Parameters
    ----------
    img : np.ndarray
        Input image (assumed float values)

    Returns
    -------
    np.ndarray
        Transformed image

    """

    if img.max() == 0:
        return img
    theta = 100 / img.max()

    return np.arcsinh(theta * img) / theta


def image_transform_root(img):
    """
    Normalise and then perform square root transform on image

    Parameters
    ----------
    img : np.ndarray
        Input image (assumed float values)

    Returns
    -------
    np.ndarray
        Transformed image

    """

    img[img < 0] = 0
    mini = img[img != 0].min()
    maxi = img.max()
    offset = (maxi - mini) / 10

    if maxi == 0 and mini == 0:
        img = img + offset
    else:
        img = (img - mini) / (maxi - mini) + offset

    return np.sqrt(img)


def image_transform_scale(img):
    """
    Small function to normalise an image between 0 and 1. Useful for deep 
    learning.

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    -------
    np.ndarray
        Scaled image

    """

    if img.min() == img.max():
        return img
    return (img - img.min()) / (img.max() - img.min())


def image_transform_resize(img, new_shape=[424, 424]):
    """
    Resize an image to new dimensions (e.g. to feed into a deep learning 
    network).

    Parameters
    ----------
    img : np.ndarray
        Input image
    new_shape : tuple
        Expected new shape for image

    Returns
    -------
    np.ndarray
        Reshaped image

    """

    return resize(img, new_shape, preserve_range=True)


def image_transform_crop(img, new_shape=[160, 160]):
    """
    Crops an image to new dimensions (assumes you want to keep the centre)

    Parameters
    ----------
    img : np.ndarray
        Input image
    new_shape : tuple
        Expected new shape for image

    Returns
    -------
    np.ndarray
        Reshaped image

    """

    delt_0 = (img.shape[0] - new_shape[0]) // 2
    delt_1 = (img.shape[1] - new_shape[1]) // 2
    return img[delt_0:img.shape[0] - delt_0, delt_1:img.shape[1] - delt_1]


def image_transform_gaussian_window(img, width=2.5):
    """
    Applies a Gaussian window of a given width to the image. This has the
    effect of downweighting possibly interfering objects near the edge of the
    image. 

    Parameters
    ----------
    img : np.ndarray
        Input image
    width : float, optional
        The standard deviation of the Gaussian. The Gaussian is evaluated on a
        grid from -5 to 5 so a width=1 corresponds to a unit Gaussian. The
        width of the Gaussian will appear to be around 1/5 of the image, which
        would be fairly aggressive downweighting of outlying sources.

    Returns
    -------
    np.ndarray
        Windowed image

    """

    xvals = np.linspace(-5, 5, img.shape[0])
    yvals = np.linspace(-5, 5, img.shape[1])
    X, Y = np.meshgrid(xvals, yvals)
    Z = 1 / np.sqrt(width) / 2 * np.exp(-(X**2 + Y**2) / 2 / width**2)

    if len(img.shape) == 2:  # Only a single channel image
        return img * Z
    else:
        new_img = np.zeros_like(img)
        for i in range(img.shape[-1]):
            new_img[:, :, i] = img[:, :, i] * Z
        return new_img


def image_transform_sigma_clipping(img, sigma=3, central=True):
    """
    Applies sigma clipping, fits contours and

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    -------
    np.ndarray

    """

    if len(img.shape) > 2:
        im = img[:, :, 0] #it uses only one channel
        #print('one channel')
    else:
        #print('ok')
        im = img

    im = np.nan_to_num(im)  # OpenCV can't handle NaNs
    #print(np.shape(im))
    mean, median, std = sigma_clipped_stats(im, sigma=sigma)
    thresh = std + median
    img_bin = np.zeros(im.shape, dtype=np.uint8)

    img_bin[im <= thresh] = 0
    img_bin[im > thresh] = 1

    contours, hierarchy = cv2.findContours(img_bin,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    x0 = img.shape[0] // 2
    y0 = img.shape[1] // 2

    for c in contours:
        if cv2.pointPolygonTest(c, (x0, y0), False) == 1:
            break

    contour_mask = np.zeros_like(img, dtype=np.uint8)
    if len(contours) == 0:
        # This happens if there's no data in the image so we just return zeros
        return contour_mask
    cv2.drawContours(contour_mask, [c], 0, (1, 1, 1), -1)

    new_img = np.zeros_like(img)
    new_img[contour_mask == 1] = img[contour_mask == 1]

    return new_img




def image_transform_greyscale(img):
    """
    Simple function that combines the rgb bands into a single image
    using OpenCVs convert colour to grayscale function.

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    -------
    np.ndarray
        Greyscale image

    """

    if len(img.shape) > 2:
        img = np.float32(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = img

    return img


def image_transform_remove_negatives(img):
    """
    Sometimes negative values (due to noise) can creep in even after sigma
    clipping which can cause problems later. Use this function before scaling
    to ensure negative values are set to zero.
    Parameters
    ----------
    img : np.ndarray
        Input image
    Returns
    -------
    np.ndarray
        Image with negatives removed

    """

    new_img = img.copy()
    new_img[new_img < 0] = 0

    return new_img


def image_transform_cv2_resize(img, scale_percent):
    """
    Function that uses OpenCVs resampling function to resize an image

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    -------
    np.ndarray
        Resized image

    """

    scale_percent = scale_percent
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def image_transform_sum_channels(img):
    """
    Small function that stacks the different channels together to form
    a new, single band image

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    -------
    np.ndarray
        Stacked image

    """

    one = img[:, :, 0]  # g-band - blue b
    two = img[:, :, 1]  # r-band - green g
    three = img[:, :, 2]  # z-band - red r

    img = np.add(one, two, three)
    return img


def image_transform_band_reorder(img):
    """
    Small function that rearranges the different channels together to form
    a new image. Made specifically for the cutout.fits files from DECALS

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    -------
    np.ndarray
        Stacked image

    """

    one = img[0, :, :]  # g-band - blue b
    two = img[1, :, :]  # r-band - green g
    three = img[2, :, :]  # z-band - red r

    img = np.dstack((three, two, one))
    return img


def image_transform_colour_correction(img, bands=('g', 'r', 'z'),
                                      scales=None, m=0.03):
    """
    Band weighting function used to match the display of astronomical images
    from the DECaLS SkyViewer and SDSS. Created specifically for DECaLS fits
    cutout files.

    Requires array shapes to contain the channel axis last in line (Default
    format for astronomical images).

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    -------
    np.ndarray
        Weighted and reordered image

    """

    rgbscales = {'u': (2, 1.5),
                 'g': (2, 6.0),
                 'r': (1, 3.4),
                 'i': (0, 1.0),
                 'z': (0, 2.2),
                 }

    if scales is not None:
        rgbscales.update(scales)

    I = 0

    for i in range(min(np.shape(img))):
        plane, scale = rgbscales[bands[i]]
        im = img[:, :, i]
        im = np.maximum(0, im * scale + m)
        I = I + im

    I /= len(bands)
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H, W = I.shape

    rgb = np.zeros((H, W, 3), np.float32)

    for i in range(min(np.shape(img))):
        plane, scale = rgbscales[bands[i]]
        im = img[:, :, i]
        rgb[:, :, plane] = (im * scale + m) * fI / I

    image = np.clip(rgb, 0, 1)
    return image


def image_transform_axis_shift(img):
    """
    Small function that shifts the band axis to the end. 
    This is used to align a fits file to the default order
    used in astronomical images.

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    -------
    np.ndarray
        Shifted image

    """

    img_channel = np.argmin(np.shape(img))

    img = np.moveaxis(img, img_channel, -1)

    return img


def image_transform_zscale(img, contrast=0.05):
    """
    Implements zscaling used in IRAF and DS9. Essentially just truncates the 
    brightness histogram at sensibly chosen minima and maxima.

    Parameters
    ----------
    img : np.ndarray
        Input image
    contrast : float, optional
        Essentially tweaks the values of z1 and z2 to be more or less 
        restrictive in the brightness distribution. 0.05 works quite well for
        MeerKAT images.

    Returns
    -------
    np.ndarray
        Shifted image

    """
    img = np.nan_to_num(img)
    zscale = ZScaleInterval(contrast=contrast)
    z1, z2 = zscale.get_limits(img)
    adj_img_arr = img.copy()
    adj_img_arr[adj_img_arr < z1] = z1
    adj_img_arr[adj_img_arr > z2] = z2
    return adj_img_arr



def image_get_first_band(img):
    return img[:,:,0]

def scaling_clipping(X):

    # Clipping and scaling parameters applied to the data as preprocessing
    vmin=-1e-9
    vmax=1e-9
    scale=100

    mask = np.where(X == 100)

    X[mask] = 0

    # Simple clipping and rescaling the images
    X = np.clip(X, vmin, vmax)/vmax * scale

    X[mask] = 0
    return X



def grayscale_average(image):
    # Convert the image to grayscale using the average method
    gray = np.mean(image, axis=2)
    return gray

def grayscale_luminosity(image):
    # Convert the image to grayscale using the luminosity method
    gray = np.dot(image, [0.21, 0.72, 0.07])
    return gray

def grayscale_lightness(image):
    # Convert the image to grayscale using the lightness method
    gray = np.max(image, axis=2)
    return gray

def grayscale_desaturation(image):
    # Convert the image to grayscale using the desaturation method
    gray = np.clip((np.max(image, axis=2) + np.min(image, axis=2)) / 2, 0, 255)
    return gray

def grayscale_single_channel(image):
    # Convert the image to grayscale using a single channel (e.g., red channel)
    gray = image[:, :, 0]
    return gray

def create_2d_rgb_image(rgb_image):
    # Compute the average value across color channels
    average_color = np.mean(rgb_image, axis=2)
    return average_color
