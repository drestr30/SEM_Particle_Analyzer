import cv2 as cv
import numpy as np
from skimage.measure import label, regionprops, regionprops_table
import argparse

def detect_and_crop(sem_img, **args):
    """
    Detects objects in an input image and crops them.

    Args:
        img (numpy.ndarray): An input image represented as a numpy array.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple of two elements.
        The first element is a listof cropped images, where each image
        contains a detected object from the input image.
        The second element is a binary mask of the input image, used to
        detect the objects.
    """
    img = remove_sem_label(sem_img, crop_h=args.get('crop_h', None))
    pixel_dx, sem_label_img = process_sem_label(sem_img, scale=args.get('scale', 50),
                                  crop_h= args.get('crop_h', None))
    mask = create_mask(img, **args)

    lbl_0 = label(mask)
    bboxes, mask = detect_from_mask(lbl_0, mask)

    props = create_props_table(lbl_0, img, pixel_dx)

    crops = []
    display = img.copy()
    for ids, box in enumerate(bboxes):
        cropped = crop_box(img, points2bbox(box))

        if False: # This is done when selecting segmentation instead of crop
            cropped_mask = crop_box(mask, points2bbox(bboxes[i]))
            cropped = crop_mask(cropped, cropped_mask, square= False)

        _ = cv.rectangle(display, (box[1], box[0]),
                         (box[3], box[2]), (255, 0, 0), 2)
        _ = cv.putText(img=display,
                       text=str(ids),
                       org=(box[1], box[0]),
                       fontFace=cv.FONT_HERSHEY_DUPLEX,
                       fontScale=2.0,
                       color=(255, 23, 55),
                       thickness=3)

        cropped = gray2rgb(cropped).astype(np.uint8)
        crops.append(cropped)

    return crops, props, {'mask' : mask,
                          'display': display,
                          'label': sem_label_img}

def create_mask(img, **args):
    mask = smooth_and_trhesh(img, threshold=args.get('threshold'))
    mask = apply_morph(mask, dilate=0, closing=0)
    return mask

def gray2rgb(gray):
    """
    Converts a 2D grayscale image to a 3D RGB image.

    Args:
        gray (numpy.ndarray): A 2D grayscale image represented as a numpy array.

    Returns:
        numpy.ndarray: A 3D RGB image represented as a numpy array with the same height and width
            as the input image, where all three color channels are set to the same value as the
            input image's pixel intensities.
    """
    h, w = gray.shape
    img = np.zeros((h, w, 3))
    img[:, :, 0] = gray
    img[:, :, 1] = gray
    img[:, :, 2] = gray
    return img


def detect_longest_horizontal_line(im):
    # if len(im.shape) < 3: # image is 3dimension
    gray = im
    image = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    # else:
    # image = im
    # gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    max_lenght = 1024
    # Apply Canny edge detection
    edges = cv.Canny(gray, 150, 300, apertureSize=3) # 50 150

    # Apply Hough line transform
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=30, maxLineGap=5)

    # Find the longest horizontal line
    longest_line_length = 0
    longest_line = None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_length = abs(x2 - x1)

        if line_length > longest_line_length and line_length < max_lenght:
            longest_line_length = line_length
            longest_line = line

    # Draw the longest line on the image (optional)
    if longest_line is not None:
        x1, y1, x2, y2 = longest_line[0]
        cv.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return longest_line_length, image

def process_sem_label(img, scale= 50, crop_h= None):
    h, w = img.shape
    if h > w:
        output = img[w:, :]
    else:
        output = img[h - 30:, :]

    if crop_h:
        off = 5
        h, w = img.shape
        output = img[h - crop_h+ off: h - off, :]

    line_lenght, label_img = detect_longest_horizontal_line(output)
    pixel_dx = scale/line_lenght
    return pixel_dx, label_img


def remove_sem_label(img, crop_h: int =None):
    """
    This function removes the label in the bottom part of the image, SEM images are original squared,
    the label is at the bottom of the image.
    
    Args:
    img (numpy.ndarray): The input image.
    
    Returns:
    numpy.ndarray: sem image without the bottom label. 
    """
    if crop_h:
        h, w = img.shape
        output = img[:h - crop_h, :]
    else: #if micrograph is square
        h, w = img.shape
        output = img[:w, :]
    return output

def smooth_and_trhesh(img, kernel=None, threshold=127):
    """
    This function applies a smoothing filter to an image and then applies a binary inverse threshold to the filtered image.
    
    Args:
    img (numpy.ndarray): The input image.
    kernel (numpy.ndarray, optional): The smoothing filter to be applied. If not provided, a 5x5 average filter will be used.
    
    Returns:
    numpy.ndarray: The binary inverse thresholded image after smoothing.
    
    Example:
    # >>> import cv2
    # >>> img = cv2.imread("image.jpg", 0)
    # >>> smooth_and_thresh(img)
    """
    if not kernel:
        # Creating the kernel with numpy
        kernel = np.ones((5, 5), np.float32)/25

    # Applying the filter
    smoothimg = cv.filter2D(src=img, ddepth=-1, kernel=kernel)
    _, smooththresh = cv.threshold(smoothimg, threshold, 255, cv.THRESH_BINARY_INV)
    return smooththresh

def apply_morph(img, dilate = 0, closing = 0 ): 
    """
    Applies morphological operations on an image using OpenCV library.

    Parameters:
        img (np.array): Input image
        dilate (int, optional): Number of iterations for dilation operation. Default is 0.
        closing (int, optional): Number of iterations for closing operation. Default is 0.

    Returns:
        np.array: The processed image.
    """
    
    open_k = np.ones((5, 5), np.uint8)
    close_k = np.ones((9,9), np.uint8)
    
    morphed = cv.morphologyEx(img, cv.MORPH_OPEN, open_k)  # opening removes small dots in the image.

    morphed = cv.dilate(morphed, open_k, iterations=dilate)  # dilatation increase mask size.

    for i in range(closing):
        morphed = cv.morphologyEx(morphed, cv.MORPH_CLOSE, close_k)
    
    return morphed

def create_props_table(lbl_0, intensity_image, dx):

    # lbl_0 = label(mask)

    def intensity_std(region, intensities):
        # note the ddof arg to get the sample var if you so desire!
        return np.std(intensities[region], ddof=1)


    props_table = regionprops_table(lbl_0, intensity_image=intensity_image,
                                    spacing= dx,
                                    properties=('intensity_mean',
                                                'solidity',
                                                'perimeter',
                                                'feret_diameter_max',
                                                'eccentricity',
                                                'area',
                                                'intensity_std',
                                                'equivalent_diameter_area',
                                                'axis_minor_length',
                                                'axis_major_length'),
                                    extra_properties=[intensity_std])
    return props_table

def detect_from_mask(lbl_0, im):
    """
    This function detects objects in an image given a binary mask. The binary mask
    is processed to obtain connected components, which are then used to draw bounding boxes
    around the objects. The function returns both the bounding boxes and the image with the
    bounding boxes drawn.
    
    Args:
    im (np.ndarray): A binary mask of the same size as the original image, with objects
                     represented as white pixels and background represented as black pixels.
                     
    Returns:
    list: A list of bounding boxes, each represented as a 4-tuple (y1, x1, y2, x2), where
          (y1, x1) and (y2, x2) are the (row, column) coordinates of the top-left and bottom-right
          corners of the bounding box, respectively.
    np.ndarray: The original image with the bounding boxes drawn.
    
    Example:
    # >>> im = np.zeros((10, 10), dtype=np.uint8)
    # >>> im[2:7, 2:7] = 255
    # >>> boxes, display = detect_from_mask(im)
    # >>> boxes
    [(2, 2, 7, 7)]
    # >>> display.shape
    (10, 10)
    """
    # lbl_0 = label(im)
    props = regionprops(lbl_0, intensity_image = im)

    display = im.copy()
    boxes = []
    for ids, prop in enumerate(props):
        _ = cv.rectangle(display, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
        _ = cv.putText(img = display,
                      text = str(ids),
                      org =  (prop.bbox[1], prop.bbox[0]),
                      fontFace = cv.FONT_HERSHEY_DUPLEX,
                      fontScale = 2.0,
                      color = (125, 246, 55),
                      thickness = 3)
        boxes.append(prop.bbox)
    
    return boxes, display

def points2bbox(points):
    """
    This function converts a set of points (y1, x1, y2, x1) representing a bounding box into the format
    (x, y, width, height), where (x, y) is the top-left corner of the bounding box.

    Args:
    points (tuple): A 4-tuple (y1, x1, y2, x2), where (y1, x1) and (y2, x2) are the 
                    (row, column) coordinates of the top-left and bottom-right corners
                    of the bounding box, respectively.

    Returns:
    tuple: A 4-tuple (x, y, width, height) representing the bounding box.
    """
    y1, x1, y2, x2 = points
    x = x1
    y = y1 
    w = x2 - x1
    h = y2 - y1
    return (x, y, w, h)

def crop_mask(sample, mask, square=True):
    """
    This function crops a sample image based on a given binary mask.
    
    Args:
    sample (numpy.ndarray): The input sample image.
    mask (numpy.ndarray): The binary mask used for cropping.
    square (bool, optional): If True, the output image will have a square shape. 
    If False, the output image will have the same shape as the mask. Default is True.
    
    Returns:
    numpy.ndarray: The cropped image.
    
    Example:
    >>> import numpy as np
    >>> sample = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    >>> crop_mask(sample, mask)
    array([[5]])
    """
    segmented = np.multiply(sample, mask)
    nzero = np.nonzero(segmented)
    top, bottom = np.min(nzero[0]), np.max(nzero[0])
    left, right = np.min(nzero[1]), np.max(nzero[1])
    
    if square:
        out = sample[top:bottom + 1, left:right + 1]
    else:
        out = segmented[top:bottom + 1, left:right + 1]
    return out

def crop_box(img, bbox):
    x, y, w, h = bbox
    crop = img[y:y+h,x:x+w]
    return crop

if __name__ == '__main__':
    path = '/media/lecun/HD/Expor2/Test images/EAFIT1 EAFIT0014.tif'
    sem_img = cv.imread(path, 0)
    particles = detect_and_crop(sem_img)
    print(len(particles[0]), 'detected particles in image')
    # for i, particle in enumerate(particles):
    #     cv.imwrite(f'.\detected\particle_{str(i)}.png', particle)