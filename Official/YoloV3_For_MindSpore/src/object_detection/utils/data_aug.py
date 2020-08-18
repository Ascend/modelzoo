import numpy as np
from PIL import Image
import random


def statistic_normalize_img(img, statistic_norm):
    # img: RGB
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = img/255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if statistic_norm:
        img = (img - mean) / std
    return img


def get_interp_method(interp, sizes=()):
    """Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and a auto-estimation method.

    Parameters
    ----------
    interp : int
        interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Nearest Neighbors. [Originally it should be Area-based,
        as we cannot find Area-based, so we use NN instead.
        Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    sizes : tuple of int
        (old_height, old_width, new_height, new_width), if None provided, auto(9)
        will return Area(2) anyway.

    Returns
    -------
    int
        interp method from 0 to 4
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            elif nh < oh and nw <ow:
                return 0
            else:
                return 1
        else:
            return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp


def PIL_image_reshape(interp):
    reshape_type = {
        0: Image.NEAREST,
        1: Image.BILINEAR,
        2: Image.BICUBIC,
        3: Image.NEAREST,
        4: Image.LANCZOS,
    }
    return reshape_type[interp]
