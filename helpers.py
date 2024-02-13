import cv2
import numpy as np

def translate_image(img,translate_dist):
    # Get the height and width of the image
    height, width = img.shape[:2]

    tx = int(translate_dist[0])
    ty = int(translate_dist[1])

    # Define the translation matrix (2x3 matrix)

    translation_matrix = np.float32([[1, 0, tx],
                                    [0, 1, ty]])

    # Apply the translation using warpAffine
    translated_image = cv2.warpAffine(img, translation_matrix, (width, height))

    return translated_image

def affine_transformation(img, affine_params):
    # Get the height and width of the image
    height, width = img.shape[:2]

    rotation_angle = int(affine_params[0])
    scaling_factor = int(affine_params[1])
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, scaling_factor)

    # Apply the affine transformation using warpAffine
    transformed_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    return transformed_image

def convert_to_grayscale(img):
    image_8u = cv2.convertScaleAbs(img)
    gray_image = cv2.cvtColor(image_8u, cv2.COLOR_RGB2GRAY)
    return gray_image

def rotate_image(img,image_rotate_angle):
    # Get image dimensions
    height, width = img.shape[:2]

    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), float(image_rotate_angle), 1)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))

    return rotated_image

def gaussian_blur(img):
    blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
    return blurred_image

def resize_image(img,new_width, new_height):
    new_width = int(new_width)
    new_height = int(new_height)

    resized_image = cv2.resize(img, (new_width, new_height))
    return resized_image

def swap_colour(image_array,image_colour_choice,image_current_colour_scheme):
    
    # Convert colour to BGR 
    if image_colour_choice == 'bgrColour':
        if image_current_colour_scheme == 'bgrColour':
            colour_swapped_image = image_array
        elif image_current_colour_scheme == 'hsvColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_HSV2BGR)
        elif image_current_colour_scheme == 'rgbColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    elif image_colour_choice == 'hsvColour':
        if image_current_colour_scheme == 'hsvColour':
            colour_swapped_image = image_array
        elif image_current_colour_scheme == 'bgrColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
        elif image_current_colour_scheme == 'rgbColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    
    elif image_colour_choice == 'rgbColour':
        if image_current_colour_scheme == 'rgbColour':
            colour_swapped_image = image_array
        elif image_current_colour_scheme == 'bgrColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        elif image_current_colour_scheme == 'hsvColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_HSV2RGB)

    return colour_swapped_image

def simple_thresh(image_array,threshold_choice,image_threshold_value,image_threshold_max):

    image_threshold_value = int(image_threshold_value)
    image_threshold_max = int(image_threshold_max)
    
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    if threshold_choice == 'binaryThresh':
        _,thresh_image = cv2.threshold(gray_image, image_threshold_value, image_threshold_max, cv2.THRESH_BINARY)
    elif threshold_choice == 'binaryThreshInv':
        _,thresh_image = cv2.threshold(gray_image, image_threshold_value, image_threshold_max, cv2.THRESH_BINARY_INV)
    elif threshold_choice == 'toZeroThresh':
        _,thresh_image = cv2.threshold(gray_image, image_threshold_value, image_threshold_max, cv2.THRESH_TOZERO)
    elif threshold_choice == 'toZeroThreshInv':
        _,thresh_image = cv2.threshold(gray_image, image_threshold_value, image_threshold_max, cv2.THRESH_TOZERO_INV)

    return thresh_image

def adapt_thresh(img, image_adaptive_parameters):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    adaptive_methods_dict = {
        'meanAdapt': cv2.ADAPTIVE_THRESH_MEAN_C,
        'gaussAdapt': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    }

    adaptive_thresholding_dict = {
        'binaryAdapt': cv2.THRESH_BINARY,
        'binaryInvAdapt': cv2.THRESH_BINARY_INV
    }

    # Convert the adaptive method string to the corresponding OpenCV constant
    adaptive_method = adaptive_methods_dict[image_adaptive_parameters[1]]
    adaptive_thresholding =  adaptive_thresholding_dict[image_adaptive_parameters[2]]

    adaptive_threshold_img = cv2.adaptiveThreshold(
        gray_image,
        int(image_adaptive_parameters[0]),  # Maximum pixel value after thresholding
        adaptive_method,  # Adaptive thresholding method
        adaptive_thresholding,  # Binary thresholding type
        int(image_adaptive_parameters[3]),  # Block size (size of the neighborhood area)
        int(image_adaptive_parameters[4])  # Constant subtracted from the mean (weighted average)
    )

    return adaptive_threshold_img



def get_hist(rgb_image_array):

    histr = []
    histr.append(cv2.calcHist([rgb_image_array], [0], None, [256], [0, 256]))
    histr.append(cv2.calcHist([rgb_image_array], [1], None, [256], [0, 256]))
    histr.append(cv2.calcHist([rgb_image_array], [2], None, [256], [0, 256]))
    return rgb_image_array, histr

def hist_equalization(rgb_image_array):
    
    equalized_img = rgb_image_array.copy()
    
    for i in range(3):
        equalized_img[:, :, i] = cv2.equalizeHist(rgb_image_array[:, :, i])
    _ , histr = get_hist(equalized_img)
    
    return equalized_img,histr

