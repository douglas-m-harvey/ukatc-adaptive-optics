# --- --- ---
# Import libraries
# --- --- ---


import numpy as np
import sympy as sy
from scipy.optimize import curve_fit

# (End of section)


# --- --- ---
# Docstring template
# --- --- ---


# Use this as a template for writing the docstrings of your functions. Anything of the form _something_ is a placeholder and should be replaced.
def func(variable_0, variable_1 = 1):
    """_summary_

    Args:
        variable_0 (_type_): _description_
        variable_1 (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if type(variable_1) is not int:
        raise Exception("variable_1 must be an int")
    output_0 = 4*variable_0
    output_1 = variable_0**variable_1
    return (output_0, output_1)

# (End of section)


# --- --- ---
# General functions
# --- --- ---


def matlab2numpy(array_matlab):
    """Convert matlab array object to numpy array.

    Args:
        array_matlab (MATLAB array object): The MATLAB array to be converted. 

    Returns:
        numpy ndarray: The resulting numpy array.
    """
    array_numpy = np.array(array_matlab._data).reshape(array_matlab.size[::-1]).T
    return array_numpy


def pad_to_shape(array, output_shape):
    """Pad an array to the given shape. 0s are appended symmetrically to all sides of the array, except when the required pad width is odd in which case the extra row and column of 0s is appended to the bottom and right of the array,
    respectively.

    Args:
        array (numpy ndarray): The input array. Must be 2-dimensional.
        output_shape (iterable of ints): Length-2 iterable defining the shape of the output padded array. Matrix-style rows by columns indexing, with the top-left corner at (0, 0).

    Returns:
        numpy ndarray: The padded array. 
    """
    y_output, x_output = output_shape
    y_input, x_input = array.shape
    y_pad = (y_output - y_input)
    x_pad = (x_output - x_input)
    return np.pad(array, ((y_pad//2, y_pad//2 + y_pad%2), (x_pad//2, x_pad//2 + x_pad%2)), mode = 'constant')


def crop_to_shape(array, output_shape):
    """Crop an array to the given shape. The array is cropped symmetrically about its centre, except when the required crop width is odd in which case the extra row and column is cropped from the bottom and right of the array,
    respectively

    Args:
        array (numpy ndarray): The input array. Must be 2-dimensional and equal to or larger than output_shape in all dimensions.
        output_shape (iterable of ints): Length-2 iterable defining the shape of the output padded array. Matrix-style rows by columns indexing, with the top-left corner at (0, 0).

    Returns:
        numpy ndarray: The cropped array.
    """
    y_output, x_output = output_shape
    y_input, x_input = array.shape
    y_crop = (y_input - y_output)
    x_crop = (x_input - x_output)
    return array.copy()[y_crop//2:y_crop//2 + y_output, x_crop//2:x_crop//2 + x_output]


def box_indices(centre, shape, limits):
    """Generate a set of indices defining a box within an array.

    Args:
        centre (iterable): Length-2 iterable defining the centre of the box. Matrix-style rows by columns indexing, with the top-left corner at (0, 0).
        shape (iterable): Length-4 iterable defining the extent of the box in each direction. Format is (top, right, bottom, left). 
        limits (iterable): Length-2 iterable defining the shape of the array in which the box is defined. If any part of the box would extend outside the array, the box is truncated to fit within it. Matrix-style rows by columns
        indexing, with the top-left corner at (0, 0).

    Returns:
        tuple of numpy ndarrays: Length 2 tuple containing 2 arrays of indices defining the box. The arrays will have shapes (n, 1) and (1, m) respectively, where n = shape[0] + shape[2] + 1 (the box's top extent + bottom extent + 1
        for the centre) and m = shape[1] + shape[3] + 1.
    """
    if type(shape) is int: shape = tuple(shape//2 for _ in range(4))
    t = centre[0] - shape[0]
    r = centre[1] + shape[1] + 1
    b = centre[0] + shape[2] + 1
    l = centre[1] - shape[3]
    if t < 0: t = 0
    if limits[1] < r: r = limits[1]
    if limits[0] < b: b = limits[0]
    if l < 0: l = 0
    return np.ix_(range(t, b), range(l, r))


def expand_box(image, centre, threshold, timeout = 64):
    """Iteratively expand a subregion in an image outwards from a central point. Each side of the box defining the subregion will expand by one pixel each iteration until one of three conditions are met: all elements along that side
    are less than or equal to the threshold value, the side reaches the edge of the image, or the number of iterations exceeds the timeout value. Each side is treated seperately, and will continue expanding until a condition is met,
    even if the others have stopped.
    This allows subregions to be defined dynamically around objects of different shapes based on a threshold that can be chosen seperately for each object. The alternative approach of simply thresholding the whole image can end up
    removing dimmer objects if other, brighter ones are present.

    Args:
        image (numpy ndarray): The input image.
        centre (iterable): Length-2 iterable defining the centre of the box. Matrix-style rows by columns indexing, with the top-left corner at (0, 0).
        threshold (float or int): The threshold value each side of the box has to be less than or equal to for that side to stop expanding. 
        timeout (int, optional): The number of iterations to run before the box is stopped from expanding any further, with each iteration expanding each side of the box that has not yet met one of the three conditions. Effectively
        controls the maximum extent of the box. Defaults to 64.

    Returns:
        tuple of numpy ndarrays: Length-2 tuple containing 2 arrays of indices defining the box. The arrays will have shapes (n, 1) and (1, m) respectively, where n is the vertical size of the box and m is its horizontal size.
    """
    expanding = np.array([True, True, True, True]) # N, E, S, W
    box_shape = np.array([1, 1, 1, 1]) # N, E, S, W
    timeout = 64
    while expanding.any():
        indices = box_indices(centre, box_shape, image.shape)
        if indices[0][0] == 0: # N
            expanding[0] = False
        if indices[1][:, -1] == image.shape[1] - 1: # E
            expanding[1] = False
        if indices[0][-1] == image.shape[0] - 1: # S
            expanding[2] = False
        if indices[1][:, 0] == 0: # W
            expanding[3] = False
        if (image[indices][0, :] > threshold).any():
            box_shape[0] += 1
        else:
            expanding[0] = False
        if (image[indices][:, -1] > threshold).any():
            box_shape[1] += 1
        else:
            expanding[1] = False
        if (image[indices][-1, :] > threshold).any():
            box_shape[2] += 1
        else:
            expanding[2] = False
        if (image[indices][:, 0] > threshold).any():
            box_shape[3] += 1
        else:
            expanding[3] = False
        timeout -= 1
        if not timeout:
            expanding = np.array([False, False, False, False])
    return indices


def gaussian_2d(x, y, A, x_0, y_0, fwhm_x, fwhm_y):
    """Return samples from a 2-dimensional Gaussian function at the specified x and y coordinates. If x and y are arrays their shapes must match.

    Args:
        x (float, int or numpy ndarray): A single x-coordinate or array of x-coordinates.
        y (float, int or numpy ndarray): A single y-coordinate or array of y-coordinates. Y-axis increases from top to bottom.
        A (float or int): Amplitude.
        x_0 (float or int): Position of the centre in x. Units defined by x and y.
        y_0 (float or int): Position of the centre in y. Units defined by x and y. Y-axis increases from top to bottom.
        fwhm_x (float or int): Full width at half maximum in x. Units defined by x and y.
        fwhm_y (float or int): Full width at half maximum in y. Units defined by x and y.

    Returns:
        float or ndarray: If x and y are floats or ints, returns the value of the Gaussian function at coordinate (x, y). If x and y are arrays, returns an array of samples from the Gaussian function at those coordinates. 
    """
    sigma_x = fwhm_x/(2*np.sqrt(2*np.log(2)))
    sigma_y = fwhm_y/(2*np.sqrt(2*np.log(2)))
    return (A*np.exp(-((x - x_0)**2/(2*sigma_x**2) + (y - y_0)**2/(2*sigma_y**2))))


def inverse_frequency_noise(power, shape, rng):
    """Generate an array of random noise with a power spectrum following a frequency**power law.

    Args:
        power (float or int): The power law for the power spectrum of the random noise to follow. 
        shape (iterable of ints): Length-2 iterable defining the shape of the output random noise array. Matrix-style rows by columns indexing, with the top-left corner at (0, 0).
        rng (numpy.random._generator.Generator): Numpy random number generator object (usualy defined as rng = numpy.random.default_rng()). If None, a new random number generator object is created.

    Returns:
        numpy ndarray: The random noise array. Scaled to the interval [-1, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    white_noise = rng.uniform(-1, 1, shape)
    white_noise_psd = np.fft.fft2(white_noise)
    y, x = np.meshgrid(np.linspace(-1, 1, white_noise.shape[1]), np.linspace(-1, 1, white_noise.shape[0]))
    distance = np.sqrt(y**2 + x**2)
    noise_psd = white_noise_psd*distance**power
    noise = np.abs(np.fft.ifft2(np.fft.fftshift(noise_psd)))
    noise -= noise.min()
    noise *= 2/noise.max()
    noise -= 1
    return noise

# (End of section)


# --- --- ---
# Centroiding functions
# --- --- ---


def centre_of_gravity(image, indices = None):
    """Calculate the centroid of an image or a subregion within an image by taking an average of the pixel coordinates weighted by their intensities.

    Args:
        image (numpy ndarray): The input image. 
        indices (iterable of numpy ndarrays, optional): Iterable containing 2 arrays of indices which define a subregion on which to calculate the centroid. Given an nxm subregion, the first array has shape (n, 1) and contains the
        vertical indices, while the second array has shape (1, m) and contains the horizontal indices. These indices are usually calculated using box_indices. If None, the centroid of the whole image is calculated. Defaults to None.

    Returns:
        numpy ndarray: Shape (2,) array containing the centroid. Matrix-style rows by columns indexing, with the top-left corner at (0, 0). 
    """
    if indices == None:
        coordinates = np.indices(image.shape)
    else:
        coordinates = np.meshgrid(*indices, indexing = "ij")
    return np.average(coordinates, (1, 2), np.tile(image, (2, 1, 1)), keepdims = False)


def gaussian_cog(image):
    """Fit a 2-dimensional Gaussian function to the image and return its centre.

    Args:
        image (numpy ndarray): The input image. 
    
    Returns:
        numpy ndarray: The centre of the fitted Gaussian function. Matrix-style rows by columns indexing, with the top-left corner at (0, 0).
    """
    def fitting_function(xy, A, x_0, y_0, fwhm_x, fwhm_y):
        # Convert 1-dimensional xy coordinates to 2-dimensional, pass to gaussian_2d, return a 1-dimensional version of the output.
        x, y = xy
        return gaussian_2d(x, y, A, x_0, y_0, fwhm_x, fwhm_y).flatten()
    xy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    max_coords = np.argwhere(np.where(image == image.max(), True, False))[0, :] # Find the coordinates of the brightest pixel in the image as an initial guess at the true centroid.
    sampling_estimate = np.sum(image >= image.max()/2)/2 # Very rough initial guess at the sampling. 
    initial_guess = (image.max(), max_coords[1], max_coords[0], sampling_estimate, sampling_estimate)
    bounds = ((0, 0, 0, np.finfo(np.float64).eps, np.finfo(np.float64).eps), (np.inf, image.shape[1], image.shape[0], np.inf, np.inf))
    popt, _ = curve_fit(fitting_function, xy, image.ravel(), p0 = initial_guess, bounds = bounds)
    return np.array([popt[2], popt[1]])

# (End of section)


# --- --- ---
# Masking functions
# --- --- ---


def circle(diameter, shape = None, centre = None):
    """Return a circular Boolean mask array.

    Args:
        diameter (float or int): The diameter of the circle in pixels
        shape (iterable of ints, optional): Length-2 iterable defining the shape of the output array. If None, the output array shape is defined to fit the diameter. Defaults to None.
        centre (iterable of floats or ints, optional): The centre of the circular mask. Matrix-style rows by columns indexing, with the top-left corner at (0, 0). Defaults to None.

    Returns:
        numpy ndarray: The circular Boolean mask array. 
    """
    shape = np.ceil(np.array([diameter, diameter])).astype(int) if shape is None else np.array(shape)
    centre = (shape - 1)/2 if centre is None else centre
    yy, xx = np.mgrid[:shape[1], :shape[0]]
    circle_distance = np.sqrt((yy - centre[0])**2 + (xx - centre[1])**2)
    circle_array = np.zeros(shape, dtype = bool)
    circle_array[(circle_distance <= diameter/2)] = True
    return circle_array


def ellipse(axes, shape = None, centre = None):
    """Return an elliptical Boolean mask array.

    Args:
        axes (iterable of floats or ints): Length-2 iterable defining the major and minor axes of the ellipse in pixels.
        shape (iterable of ints, optional): Length-2 iterable defining the shape of the output array. If None, the output array shape is defined to fit the diameter. Defaults to None.
        centre (iterable of floats or ints, optional): The centre of the elliptical mask. Matrix-style rows by columns indexing, with the top-left corner at (0, 0). Defaults to None.

    Returns:
        numpy ndarray: The elliptical Boolean mask array. 
    """
    shape = np.ceil(np.array(axes)).astype(int) if shape is None else np.array(shape)
    centre = (shape - 1)/2 if centre is None else centre
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    ellipse_distance = np.sqrt((yy - centre[0])**2/(axes[0]/2)**2 + (xx - centre[1])**2/(axes[1]/2)**2)
    ellipse_array = np.zeros(shape, dtype = bool)
    ellipse_array[(ellipse_distance <= 1)] = True
    return ellipse_array

# (End of section)


# --- --- ---
# Zernike functions
# --- --- ---


def noll(index, return_list = False):
    """Convert Noll Zernike mode indices to nxm radial and azimuthal indices.

    Args:
        index (int): The Noll index to convert.
        return_list (bool, optional): Whether or not to return a list of all nxm indices before and including the ones corresponding the given Noll index. Defaults to False.

    Returns:
        tuple or nested lists of tuples: If return_list is False, returns a length-2 tuple containing the n and m indices corresponding to the given Noll index. If return_list is True, returns a list of lists, with each sublist
        containing a tuple for each pair of nxm indices for one radial order (e.g. with n constant), for a number of radial orders equal to the Noll index. For example: 
        "noll(4, return_list=True)"
        "[[(0, 0)], [(1, 1), (1, -1)], [(2, 0), (2, -2), (2, 2)], [(3, -1), (3, -3), (3, 1), (3, 3)]]"
        
        I don't know why it works this way, but I'm sure I had a reason for writing it like this at the time. :)
    """
    noll_dict = {n : [m for m in np.arange(-n, n + 1, 2)] for n in np.arange(index)}
    odd_count = 0
    for n in noll_dict:
        if n%2 == 0:
            sorted_list = []
            ind = n//2
            for alt in range(n + 1):
                ind += alt*(-1)**alt
                sorted_list.append(noll_dict[n][ind])
            noll_dict[n] = sorted_list
        if n%2 != 0:
            sorted_list = []
            if odd_count%2 == 0:
                ind = n//2 + 1
            else:
                ind = n//2
            for alt in range(n + 1):
                ind += alt*(-1)**alt
                sorted_list.append(noll_dict[n][ind])
            noll_dict[n] = sorted_list
            odd_count += 1
    noll_list = [[(n, m) for m in noll_dict[n]] for n in range(index)]
    if not return_list:
        flattened_noll_list = [nm_index for sublist in noll_list for nm_index in sublist]
        return flattened_noll_list[index - 1]
    return noll_list


def zernikes(indices, indexing_type = "noll", normalised = True):
    """Take a list of Zernike mode indices and generate a function which takes polar coordinate arrays rho and phi and an iterable of the coefficients of the modes and returns an array of the weighted sum of the modes defined over
    those coordinates. This function calculates analytically the sum of the specified modes, which can take some time, then converts the resulting expression into a highly-optimised python function which executes quickly regardless of
    the pupil resolution. 
    
    For example, to generate a function which returns the sum of the first 6 modes:
    "zernike_sum_function = zernikes([1, 2, 3, 4, 5, 6], indexing_type = "noll")"
    Then generate an array which is the sum of those modes over a 128x128 pixel pupil, with the coefficient of each successive mode incremented by 3:
    "pupil = circle(128)"
    "x, y = np.meshgrid(np.linspace(-1, 1, pupil.shape[0]), np.linspace(-1, 1, pupil.shape[1]))"
    "rho = np.sqrt(x**2 + y**2)"
    "phi = np.arctan2(y, x)"
    "zernike_sum = zernike_sum_function(rho, phi, [1, 4, 7, 10, 13, 15])"

    Args:
        indices (iterable of ints or length-2 iterables): The indices specifying the modes to include in the sum. If indexing_type is "noll", this should be an iterable of ints. If indexing_type is "nm", this should be an iterable of
        length-2 iterables representing the azimuthal and radial indices of the modes.
        indexing_type (str, optional): Specifies the indexing scheme to use. Can be either "noll" or "nm". Defaults to "noll".
        normalised (bool, optional): Specifies whether or not to normalise the modes by their RMS, so that RMS(c*mode) = c. Defaults to True.

    Returns:
        function: The resulting function. Takes polar coordinate arrays rho and phi and an iterable containing a coefficient for each mode (coefficients can be 0). Returns the sum of the modes in an array with shape equal to the shape
        of the coordinate arrays.
    """
    indexing_type = indexing_type.lower()
    modes = []
    coefficients = []
    for index in indices:
        if indexing_type == "noll":
            n, l = noll(index)
        elif indexing_type == "nm":
            n, l = index
        m = np.abs(l)
        k, rho, phi = sy.symbols("k, rho, phi")
        a = sy.symbols(f"a_{index}")
        coefficients.append(a)
        R_n_m = ((-1)**k*sy.factorial(n - k)/(sy.factorial(k)*sy.factorial((n + m)//2 - k)*sy.factorial((n - m)//2 - k)))*rho**(n - 2*k)
        R = a*sy.Piecewise((sy.Sum(R_n_m, (k, 0, (n - m)//2)), sy.simplify(n - m).is_even), (0, sy.simplify(n - m).is_odd))
        if l >= 0:
            mode = R*sy.cos(m*phi)
        elif l < 0:
            mode = R*sy.sin(m*phi)
        if normalised:
            mode /= sy.sqrt((sy.integrate((mode/a)**2*rho, (rho, 0, 1), (phi, -sy.pi, sy.pi))/sy.pi))
        modes.append(mode)
    zernike_lambdified = sy.lambdify([rho, phi, coefficients], sum(modes).doit())
    return zernike_lambdified

# (End of section)