import cv2
import numpy as np
from math import sqrt,exp
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import torch
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import base64
import imageio
import pytesseract
from pdf2image import convert_from_path
import requests
import pickle
import xgboost as xgb

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelEncoder
import mediapipe as mp

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the model files
custom_model_path = os.path.join(current_dir, 'models', 'custom.h5')
vgg16_model_path = os.path.join(current_dir, 'models', 'vgg16.h5')
resnet_model_path = os.path.join(current_dir, 'models', 'resnet.h5')
best_model_path = os.path.join(current_dir,'models','best.pt')
fasterrccn_model_path = os.path.join(current_dir,'models','fasterrcnn_resnet50_fpn.pth')

letters_dict_asl = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}


def translate_image(img, translate_dist):
    """
    Translate the given image by specified distances along the x and y axes.

    Parameters:
    img (numpy.ndarray): The input image to be translated.
    translate_dist (str): The translation distances in the format 'tx,ty' where
                          tx is the translation distance along the x-axis and
                          ty is the translation distance along the y-axis.

    Returns:
    numpy.ndarray: The translated image.
    """
    # Get the height and width of the image
    height, width = img.shape[:2]
    translate_dist = translate_dist.split(',')
    tx = int(translate_dist[0])
    ty = int(translate_dist[1])

    # Define the translation matrix (2x3 matrix)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    # Apply the translation using warpAffine
    translated_image = cv2.warpAffine(img, translation_matrix, (width, height))

    return translated_image

def affine_transformation(img, affine_params):
    """
    Apply an affine transformation to the given image using specified rotation angle and scaling factor.

    Parameters:
    img (numpy.ndarray): The input image to be transformed.
    affine_params (str): The affine transformation parameters in the format 'rotation_angle,scaling_factor' 
                         where rotation_angle is the angle of rotation in degrees and scaling_factor is 
                         the factor by which the image is scaled.

    Returns:
    numpy.ndarray: The transformed image.
    """
    # Get the height and width of the image
    height, width = img.shape[:2]
    affine_params = affine_params.split(',')
    rotation_angle = int(affine_params[0])
    scaling_factor = float(affine_params[1])
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, scaling_factor)

    # Apply the affine transformation using warpAffine
    transformed_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    return transformed_image

def convert_to_grayscale(img):
    """
    Convert the given image to grayscale.

    Parameters:
    img (numpy.ndarray): The input image to be converted.

    Returns:
    numpy.ndarray: The grayscale image.
    """
    image_8u = cv2.convertScaleAbs(img)
    gray_image = cv2.cvtColor(image_8u, cv2.COLOR_RGB2GRAY)
    return gray_image

def rotate_image(img, image_rotate_angle):
    """
    Rotate the given image by a specified angle.

    Parameters:
    img (numpy.ndarray): The input image to be rotated.
    image_rotate_angle (float): The angle by which to rotate the image.

    Returns:
    numpy.ndarray: The rotated image.
    """
    # Get image dimensions
    height, width = img.shape[:2]

    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), float(image_rotate_angle), 1)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))

    return rotated_image

def resize_image(img, new_width, new_height):
    """
    Resize the given image to specified dimensions.

    Parameters:
    img (numpy.ndarray): The input image to be resized.
    new_width (int): The desired width of the resized image.
    new_height (int): The desired height of the resized image.

    Returns:
    numpy.ndarray: The resized image.
    """
    new_width = int(new_width)
    new_height = int(new_height)

    resized_image = cv2.resize(img, (new_width, new_height))
    return resized_image




def reconstruct_image(pixel_data, image_width, image_height, input_format, output_format):
    """
    Reconstruct and convert the image based on the provided pixel data and format specifications.

    Parameters:
    pixel_data (numpy.ndarray): A numpy array of shape (height, width, 3) containing pixel data.
    image_width (int): The width of the image.
    image_height (int): The height of the image.
    input_format (str): The format of the input image (e.g., 'rgbColour', 'bgrColour', 'hsvColour').
    output_format (str): The desired format of the output image (e.g., 'rgbColour', 'bgrColour', 'hsvColour').

    Returns:
    numpy.ndarray: The reconstructed and converted image.
    """

    # Ensure the input image dimensions match the provided pixel data shape
    if pixel_data.shape != (image_height, image_width, 3):
        raise ValueError("The shape of pixel_data does not match the specified image dimensions.")

    # Handle special case for 'hsvColour' input format
    if input_format == 'hsvColour':
        input_format = 'rgbColour'

    # Convert the image from the input format to the desired output format
    if output_format == 'bgrColour':
        if input_format == 'bgrColour':
            colour_swapped_image = pixel_data
        elif input_format == 'hsvColour':
            colour_swapped_image = cv2.cvtColor(pixel_data, cv2.COLOR_HSV2BGR)
        elif input_format == 'rgbColour':
            colour_swapped_image = cv2.cvtColor(pixel_data, cv2.COLOR_RGB2BGR)
    elif output_format == 'hsvColour':
        if input_format == 'hsvColour':
            colour_swapped_image = pixel_data
        elif input_format == 'bgrColour':
            colour_swapped_image = cv2.cvtColor(pixel_data, cv2.COLOR_BGR2HSV)
        elif input_format == 'rgbColour':
            colour_swapped_image = cv2.cvtColor(pixel_data, cv2.COLOR_RGB2HSV)
    elif output_format == 'rgbColour':
        if input_format == 'rgbColour':
            colour_swapped_image = pixel_data
        elif input_format == 'bgrColour':
            colour_swapped_image = cv2.cvtColor(pixel_data, cv2.COLOR_BGR2RGB)
        elif input_format == 'hsvColour':
            colour_swapped_image = cv2.cvtColor(pixel_data, cv2.COLOR_HSV2RGB)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    return colour_swapped_image


def swap_colour(image_array, image_colour_choice, image_current_colour_scheme):
    """
    Swap the colour space of the given image to the specified colour space.

    Parameters:
    image_array (numpy.ndarray): The input image array to be colour swapped.
    image_colour_choice (str): The desired colour space ('bgrColour', 'hsvColour', 'rgbColour').
    image_current_colour_scheme (str): The current colour space of the image ('bgrColour', 'hsvColour', 'rgbColour').

    Returns:
    numpy.ndarray: The image with the colour space swapped to the desired colour space.
    """
    # Convert colour to BGR
    if image_colour_choice == 'bgrColour':
        if image_current_colour_scheme == 'bgrColour':
            colour_swapped_image = image_array
        elif image_current_colour_scheme == 'hsvColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_HSV2BGR)
        elif image_current_colour_scheme == 'rgbColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Convert colour to HSV
    elif image_colour_choice == 'hsvColour':
        if image_current_colour_scheme == 'hsvColour':
            colour_swapped_image = image_array
        elif image_current_colour_scheme == 'bgrColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
        elif image_current_colour_scheme == 'rgbColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    # Convert colour to RGB
    elif image_colour_choice == 'rgbColour':
        if image_current_colour_scheme == 'rgbColour':
            colour_swapped_image = image_array
        elif image_current_colour_scheme == 'bgrColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        elif image_current_colour_scheme == 'hsvColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_HSV2RGB)

    return colour_swapped_image

def simple_thresh(image_array, threshold_choice, image_threshold_value, image_threshold_max):
    """
    Apply a simple threshold to the given image based on the specified threshold type and values.

    Parameters:
    image_array (numpy.ndarray): The input image array to be thresholded.
    threshold_choice (str): The type of threshold to apply ('binaryThresh', 'binaryThreshInv', 'toZeroThresh', 'toZeroThreshInv').
    image_threshold_value (int): The threshold value.
    image_threshold_max (int): The maximum value to use with the thresholding.

    Returns:
    numpy.ndarray: The thresholded image.
    """
    image_threshold_value = int(image_threshold_value)
    image_threshold_max = int(image_threshold_max)
    
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    if threshold_choice == 'binaryThresh':
        _, thresh_image = cv2.threshold(gray_image, image_threshold_value, image_threshold_max, cv2.THRESH_BINARY)
    elif threshold_choice == 'binaryThreshInv':
        _, thresh_image = cv2.threshold(gray_image, image_threshold_value, image_threshold_max, cv2.THRESH_BINARY_INV)
    elif threshold_choice == 'toZeroThresh':
        _, thresh_image = cv2.threshold(gray_image, image_threshold_value, image_threshold_max, cv2.THRESH_TOZERO)
    elif threshold_choice == 'toZeroThreshInv':
        _, thresh_image = cv2.threshold(gray_image, image_threshold_value, image_threshold_max, cv2.THRESH_TOZERO_INV)

    return thresh_image

def adapt_thresh(img, image_adaptive_parameters):
    """
    Apply adaptive thresholding to the given image using specified parameters.

    Parameters:
    img (numpy.ndarray): The input image to be thresholded.
    image_adaptive_parameters (str): Comma-separated string of adaptive thresholding parameters:
                                     'max_pixel_value,adaptive_method,thresholding_type,block_size,C'.
                                     - max_pixel_value (int): The maximum pixel value after thresholding.
                                     - adaptive_method (str): The adaptive method to use ('meanAdapt', 'gaussAdapt').
                                     - thresholding_type (str): The thresholding type ('binaryAdapt', 'binaryInvAdapt').
                                     - block_size (int): Size of the neighborhood area.
                                     - C (int): Constant subtracted from the mean (weighted average).

    Returns:
    numpy.ndarray: The adaptive thresholded image.
    """
    # Split the string into a list
    parameters = image_adaptive_parameters.split(',')

    # Extract individual parameters from the list
    max_pixel_value = int(parameters[0])
    adaptive_method_str = parameters[1]
    thresholding_type_str = parameters[2]
    block_size = int(parameters[3])
    C = int(parameters[4])

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
    adaptive_method = adaptive_methods_dict[adaptive_method_str]
    adaptive_thresholding = adaptive_thresholding_dict[thresholding_type_str]

    adaptive_threshold_img = cv2.adaptiveThreshold(
        gray_image,
        max_pixel_value,  # Maximum pixel value after thresholding
        adaptive_method,  # Adaptive thresholding method
        adaptive_thresholding,  # Binary thresholding type
        block_size,  # Block size (size of the neighborhood area)
        C  # Constant subtracted from the mean (weighted average)
    )

    return adaptive_threshold_img

def otsu_thresh(img, image_threshold_value, image_threshold_max):
    """
    Apply Otsu's thresholding to the given image.

    Parameters:
    img (numpy.ndarray): The input image to be thresholded.
    image_threshold_value (int): The threshold value (ignored in Otsu's method).
    image_threshold_max (int): The maximum value to use with the thresholding.

    Returns:
    numpy.ndarray: The Otsu thresholded image.
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, otsu_thresh_image = cv2.threshold(gray_image, int(image_threshold_value), int(image_threshold_max), cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return otsu_thresh_image

def get_hist(rgb_image_array):
    """
    Calculate the histogram for each color channel (R, G, B) of the given image.

    Parameters:
    rgb_image_array (numpy.ndarray): The input RGB image array.

    Returns:
    tuple: The original RGB image array and a list of histograms for each color channel.
    """
    histr = []
    histr.append(cv2.calcHist([rgb_image_array], [0], None, [256], [0, 256]))
    histr.append(cv2.calcHist([rgb_image_array], [1], None, [256], [0, 256]))
    histr.append(cv2.calcHist([rgb_image_array], [2], None, [256], [0, 256]))
    return rgb_image_array, histr

def hist_equalization(rgb_image_array):
    """
    Apply histogram equalization to each color channel of the given RGB image.

    Parameters:
    rgb_image_array (numpy.ndarray): The input RGB image array.

    Returns:
    numpy.ndarray: The histogram-equalized image.
    """
    ycrcb_image = cv2.cvtColor(rgb_image_array, cv2.COLOR_RGB2YCrCb)
    ycrcb_image[:, :, 0] = cv2.equalizeHist(ycrcb_image[:, :, 0])
    equalized_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2RGB)
    
    _, hist = get_hist(equalized_image)
    
    return equalized_image, hist

def smooth_kernel(img, image_selected_kernel):
    """
    Apply a smoothing filter to the given image based on the selected kernel.

    Parameters:
    img (numpy.ndarray): The input image to be smoothed.
    image_selected_kernel (str): The type of smoothing kernel to apply ('boxKernel', 'gaussianKernel', 'medianKernel', 'bilateralKernel').

    Returns:
    numpy.ndarray: The smoothed image.
    """
    if image_selected_kernel == 'boxKernel':
        kernel_size = (5, 5)
        # Apply the box filter
        smoothed_image = cv2.boxFilter(img, -1, kernel_size)
    elif image_selected_kernel == 'gaussianKernel':
        kernel_size = (5, 5)
        smoothed_image = cv2.GaussianBlur(img, kernel_size, 0)
    elif image_selected_kernel == 'medianKernel':
        kernel_size = 5
        # Apply the median filter
        smoothed_image = cv2.medianBlur(img, kernel_size)
    elif image_selected_kernel == 'bilateralKernel':
        diameter = 9
        sigma_color = 75
        sigma_space = 75
        # Apply the bilateral filter
        smoothed_image = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)

    return smoothed_image

def edge_kernel(img, image_selected_kernel):
    """
    Apply an edge detection filter to the given image based on the selected kernel.

    Parameters:
    img (numpy.ndarray): The input image for edge detection.
    image_selected_kernel (str): The type of edge detection kernel to apply ('sobelXKernel', 'sobelYKernel', 'sobelCKernel', 'prewittXKernel', 'prewittYKernel', 'prewittCKernel', 'scharrXKernel', 'scharrYKernel', 'scharrCKernel').

    Returns:
    numpy.ndarray: The image with edges detected.
    """
    image_8u = cv2.convertScaleAbs(img)
    gray_image = cv2.cvtColor(image_8u, cv2.COLOR_RGB2GRAY)

    if image_selected_kernel == 'sobelXKernel':
        edge_image = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    elif image_selected_kernel == 'sobelYKernel':
        edge_image = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    elif image_selected_kernel == 'sobelCKernel':
        # Apply Sobel operator in X direction
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)
        # Apply Sobel operator in Y direction
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = cv2.convertScaleAbs(sobel_y)
        # Combine Sobel X and Y results
        edge_image = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    elif image_selected_kernel == 'prewittXKernel':
        # Define the PrewittX kernel
        prewitt_x_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        # Apply the PrewittX kernel using cv2.filter2D()
        edge_image = cv2.filter2D(gray_image, cv2.CV_64F, prewitt_x_kernel)
    elif image_selected_kernel == 'prewittYKernel':
        # Define the PrewittY kernel
        prewitt_y_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        # Apply the PrewittY kernel using cv2.filter2D()
        edge_image = cv2.filter2D(gray_image, cv2.CV_64F, prewitt_y_kernel)
    elif image_selected_kernel == 'prewittCKernel':
        # Define Prewitt kernels
        prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        # Apply Prewitt operator in X direction
        prewitt_x = cv2.filter2D(gray_image, -1, prewitt_kernel_x)
        prewitt_x = cv2.convertScaleAbs(prewitt_x)
        # Apply Prewitt operator in Y direction
        prewitt_y = cv2.filter2D(gray_image, -1, prewitt_kernel_y)
        prewitt_y = cv2.convertScaleAbs(prewitt_y)
        # Combine Prewitt X and Y results
        edge_image = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
    elif image_selected_kernel == 'scharrXKernel':
        scharr_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
        edge_image = cv2.convertScaleAbs(scharr_x)
    elif image_selected_kernel == 'scharrYKernel':
        scharr_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
        edge_image = cv2.convertScaleAbs(scharr_y)
    elif image_selected_kernel == 'scharrCKernel':
        # Apply Scharr operator in X direction
        scharr_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
        scharr_x = cv2.convertScaleAbs(scharr_x)
        # Apply Scharr operator in Y direction
        scharr_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
        scharr_y = cv2.convertScaleAbs(scharr_y)
        # Combine Scharr X and Y results
        edge_image = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)

    return edge_image

def sharp_kernel(img, image_selected_kernel):
    """
    Apply a sharpening filter to the given image based on the selected kernel.

    Parameters:
    img (numpy.ndarray): The input image to be sharpened.
    image_selected_kernel (str): The type of sharpening kernel to apply ('basicSharpKernel', 'laplaSharpKernel', 'unSharpKernel').

    Returns:
    numpy.ndarray: The sharpened image.
    """
    if image_selected_kernel == 'basicSharpKernel':
        # Define a basic sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # Apply the sharpening kernel to the image
        sharp_img = cv2.filter2D(img, -1, kernel)
    elif image_selected_kernel == 'laplaSharpKernel':
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply the Laplacian filter to detect edges
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        # Convert the single-channel Laplacian to a 3-channel image
        laplacian_3channel = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        # Sharpen the image by adding the Laplacian to the original image
        sharp_img = cv2.addWeighted(img, 1.0, laplacian_3channel, 1.0, 0)
    elif image_selected_kernel == 'unSharpKernel':
        # Apply a Gaussian blur to the image
        blurred = cv2.GaussianBlur(img, (9, 9), 10.0)
        # Create the unsharp mask
        sharp_img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    return sharp_img

def custom_kernel(img, provided_kernel):
    """
    Apply a custom kernel to the given image.

    Parameters:
    img (numpy.ndarray): The input image to be processed.
    provided_kernel (str): Comma-separated string representing the custom kernel values.

    Returns:
    numpy.ndarray: The image processed with the custom kernel.
    """
    # Convert the provided kernel string to a list of floats
    kernel_list = list(map(float, provided_kernel.split(',')))
    # Determine the size of the kernel
    kernel_size = int(np.sqrt(len(kernel_list)))
    # Reshape the list into a 2D numpy array
    kernel_array = np.array(kernel_list).reshape((kernel_size, kernel_size))

    # Apply the custom kernel to the image
    new_image = cv2.filter2D(img, -1, kernel_array)

    return new_image

def dilate_image(img, morph_selection):
    """
    Apply dilation to the given image using the selected morphological operation.

    Parameters:
    img (numpy.ndarray): The input

 image to be dilated.
    morph_selection (str): The type of morphological operation to apply ('dilate', 'erode', 'open', 'close', 'gradient', 'tophat', 'blackhat').

    Returns:
    numpy.ndarray: The image after applying the morphological operation.
    """
    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    if morph_selection == 'dilate':
        result_image = cv2.dilate(img, kernel, iterations=1)
    elif morph_selection == 'erode':
        result_image = cv2.erode(img, kernel, iterations=1)
    elif morph_selection == 'open':
        result_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif morph_selection == 'close':
        result_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif morph_selection == 'gradient':
        result_image = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    elif morph_selection == 'tophat':
        result_image = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    elif morph_selection == 'blackhat':
        result_image = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    return result_image

def get_contours(img):
    """
    Convert the given image to grayscale, apply GaussianBlur to reduce noise, 
    threshold the image, and find contours in the binary image.

    Parameters:
    img (numpy.ndarray): The input image from which to find contours.

    Returns:
    list: A list of contours found in the image.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # Find contours in the binary image
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cnts

def draw_contours(img):
    """
    Draw all contours on the given image.

    Parameters:
    img (numpy.ndarray): The input image on which to draw contours.

    Returns:
    numpy.ndarray: The image with contours drawn.
    """
    cnts = get_contours(img)
    # Draw all contours on the original image
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 3)

    return img

def draw_area(img, cnts):
    """
    Draw contours and annotate them with their respective areas on the given image.

    Parameters:
    img (numpy.ndarray): The input image on which to draw contours and annotate areas.
    cnts (list): A list of contours to draw and annotate.

    Returns:
    numpy.ndarray: The image with contours and area annotations drawn.
    """
    for i, contour in enumerate(cnts):
        area = cv2.contourArea(contour)

        # Draw contour on the image
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

        # Add text annotation with contour area
        cv2.putText(img, f"{i+1}: {round(area, 2)}", (int(contour[0][0][0]), int(contour[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

def draw_perimeter(img, cnts):
    """
    Draw contours and annotate them with their respective perimeters on the given image.

    Parameters:
    img (numpy.ndarray): The input image on which to draw contours and annotate perimeters.
    cnts (list): A list of contours to draw and annotate.

    Returns:
    numpy.ndarray: The image with contours and perimeter annotations drawn.
    """
    for i, contour in enumerate(cnts):
        perimeter = cv2.arcLength(contour, True)

        # Draw contour on the image
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

        # Add text annotation with contour perimeter
        cv2.putText(img, f"{i+1}: {round(perimeter, 2)}", (int(contour[0][0][0]), int(contour[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

def draw_centre(img, cnts):
    """
    Draw the centroids of the given contours on the image.

    Parameters:
    img (numpy.ndarray): The input image on which to draw centroids.
    cnts (list): A list of contours for which to find and draw centroids.

    Returns:
    numpy.ndarray: The image with centroids drawn.
    """
    for i, contour in enumerate(cnts):
        moments = cv2.moments(contour)

        # Calculate centroid coordinates
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = 0, 0

        # Draw contour on the image
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

        # Draw a dot at the centroid
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    return img

def show_contour_properties(img, selected_property):
    """
    Show specific contour properties (area, perimeter, or centroid) on the given image.

    Parameters:
    img (numpy.ndarray): The input image on which to show contour properties.
    selected_property (str): The contour property to show ('contourArea', 'contourPerimeter', 'contourCentre').

    Returns:
    numpy.ndarray: The image with the selected contour properties shown.
    """
    cnts = get_contours(img)

    if selected_property == 'contourArea':
        img = draw_area(img, cnts)
    elif selected_property == 'contourPerimeter':
        img = draw_perimeter(img, cnts)
    elif selected_property == 'contourCentre':
        img = draw_centre(img, cnts)
    
    return img

def draw_bounding_rectangle(img, cnts):
    """
    Draw bounding rectangles around the given contours on the image.

    Parameters:
    img (numpy.ndarray): The input image on which to draw bounding rectangles.
    cnts (list): A list of contours for which to draw bounding rectangles.

    Returns:
    numpy.ndarray: The image with bounding rectangles drawn.
    """
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img

def draw_bounding_rot_rectangle(img, cnts):
    """
    Draw rotated bounding rectangles around the given contours on the image.

    Parameters:
    img (numpy.ndarray): The input image on which to draw rotated bounding rectangles.
    cnts (list): A list of contours for which to draw rotated bounding rectangles.

    Returns:
    numpy.ndarray: The image with rotated bounding rectangles drawn.
    """
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    return img 

def draw_bounding_circle(img, cnts):
    """
    Draw bounding circles around the given contours on the image.

    Parameters:
    img (numpy.ndarray): The input image on which to draw bounding circles.
    cnts (list): A list of contours for which to draw bounding circles.

    Returns:
    numpy.ndarray: The image with bounding circles drawn.
    """
    for cnt in cnts:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(img, center, radius, (0, 255, 0), 2)
    
    return img

def draw_bounding_ellipse(img, cnts):
    """
    Draw bounding ellipses around the given contours on the image.

    Parameters:
    img (numpy.ndarray): The input image on which to draw bounding ellipses.
    cnts (list): A list of contours for which to draw bounding ellipses.

    Returns:
    numpy.ndarray: The image with bounding ellipses drawn.
    """
    for cnt in cnts:
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)

    return img

def show_contour_bounding_box(img, bb_selection):
    """
    Draw bounding boxes around contours on the given image based on the selected bounding box type.

    Parameters:
    img (numpy.ndarray): The input image on which to draw bounding boxes.
    bb_selection (str): The type of bounding box to draw ('boundingRectangle', 'boundingOrRectangle', 'boundingCircle', 'boundingEllipse').

    Returns:
    numpy.ndarray: The image with bounding boxes drawn.
    """
    cnts = get_contours(img)

    if bb_selection == 'boundingRectangle':
        img = draw_bounding_rectangle(img, cnts)
    elif bb_selection == 'boundingOrRectangle':
        img = draw_bounding_rot_rectangle(img, cnts)
    elif bb_selection == 'boundingCircle':
        img = draw_bounding_circle(img, cnts)
    elif bb_selection == 'boundingEllipse':
        img = draw_bounding_ellipse(img, cnts)
    
    return img

def identify_shapes(img):
    """
    Identify and label geometric shapes in the given image based on contour detection.

    Parameters:
    img (numpy.ndarray): The input image in which to identify shapes.

    Returns:
    numpy.ndarray: The image with identified shapes labeled.
    """
    # Converting image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Using GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Using Canny edge detection
    edged = cv2.Canny(blurred, 50, 150)

    # Finding contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours
    for contour in contours:
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        vertices = len(approx)

        # Draw the contour
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)

        # Find the center of the shape
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            continue

        # Identify the shape based on the number of vertices
        if vertices == 3:
            shape_name = "Triangle"
        elif vertices == 4:
            # Calculate aspect ratio to distinguish between square and rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
        elif vertices == 5:
            shape_name = "Pentagon"
        elif vertices == 6:
            shape_name = "Hexagon"
        else:
            # Use the area and the arc length to approximate a circle
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.8:
                shape_name = "Circle"
            else:
                shape_name = "Polygon"

        # Put the name of the shape at the center
        cv2.putText(img, shape_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img

def calculate_2dft(input):
    """
    Compute the 2-dimensional Fourier Transform of the given input image.

    Parameters:
    input (numpy.ndarray): The input image for which to compute the Fourier Transform.

    Returns:
    numpy.ndarray: The Fourier Transformed image.
    """
    ft = np.fft.fft2(input)
    ft = np.fft.fftshift(ft)
    return ft

def high_pass_filter_fft(ft, cutoff_frequency):
    """
    Apply a high-pass filter to the Fourier transformed image.

    Parameters:
    ft (numpy.ndarray): The Fourier Transformed image.
    cutoff_frequency (int): The cutoff frequency for the high-pass filter.

    Returns:
    numpy.ndarray: The high-pass filtered Fourier Transformed image.
    """
    rows, cols = ft.shape
    crow, ccol = rows // 2, cols // 2  # Center
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow)**2 + (j - ccol)**2) >= cutoff_frequency:
                mask[i, j] = 1
                
    # Apply the high-pass filter directly to the Fourier transformed data
    fft_filtered = ft * mask
    return fft_filtered

def low_pass_filter_fft(ft, cutoff_frequency):
    """
    Apply a low-pass filter to the Fourier transformed image.

    Parameters:
    ft (numpy.ndarray): The Fourier Transformed image.
    cutoff_frequency (int): The cutoff frequency for the low-pass filter.

    Returns:
    numpy.ndarray: The low-pass filtered Fourier Transformed image.
    """
    rows, cols = ft.shape
    crow, ccol = rows // 2, cols // 2  # Center
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow)**2 + (j - ccol)**2) <= cutoff_frequency:
                mask[i, j] = 1
                
    # Apply the low-pass filter directly to the Fourier transformed data
    fft_filtered = ft * mask
    return fft_filtered

def gaussian_high_pass_filter_fft(ft, cutoff_frequency):
    """
    Apply a Gaussian high-pass filter to the Fourier transformed image.

    Parameters:
    ft (numpy.ndarray): The Fourier Transformed image.
    cutoff_frequency (int): The cutoff frequency for the Gaussian high-pass filter.

    Returns:
    numpy.ndarray: The high-pass filtered Fourier Transformed image.
    """
    rows, cols = ft.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            mask[i, j] = 1 - np.exp(-(distance**2) / (2 * (cutoff_frequency**2)))
            
    # Apply the Gaussian high-pass filter
    fft_filtered = ft * mask
    return fft_filtered

def gaussian_low_pass_filter_fft(ft, cutoff_frequency):
    """
    Apply a Gaussian low-pass filter to the Fourier transformed image.

    Parameters:
    ft (numpy.ndarray): The Fourier Transformed image.
    cutoff_frequency (int): The cutoff frequency for the Gaussian low-pass filter.

    Returns:
    numpy.ndarray: The low-pass filtered Fourier Transformed image.
    """
    rows, cols = ft.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            mask[i, j] = np.exp(-(distance**2) / (2 * (cutoff_frequency**2)))
            
    # Apply the Gaussian low-pass filter
    fft_filtered = ft * mask
    return fft_filtered

def fourrier(img):
    """
    Convert the given image to grayscale and compute its 2-dimensional Fourier Transform.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The Fourier Transformed grayscale image.
    """
    # Convert to grayscale
    gray_img = convert_to_grayscale(img)
    ft = calculate_2dft(gray_img)

    return ft

def inverse_Fourier(ft):
    """
    Compute the inverse Fourier Transform of the given Fourier transformed image.

    Parameters:
    ft (numpy.ndarray): The Fourier Transformed image.

    Returns:
    numpy.ndarray: The image obtained after applying the inverse Fourier Transform.
    """
    ift = np.fft.ifftshift(ft)
    ift = np.fft.ifft2(ift)  
    ift = ift.real  
    return ift

def fourier_spectrum_20(img):
    """
    Compute the Fourier Transform of the given image and return its magnitude spectrum.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The magnitude spectrum of the Fourier Transformed image.
    """
    # Convert to grayscale
    gray_img = convert_to_grayscale(img)
    ft = calculate_2dft(gray_img)

    magnitude_spectrum = 20 * np.log(np.abs(ft))

    # Convert the magnitude spectrum to a format that can be displayed
    # Normalize to fit [0, 255] and convert to uint8
    magnitude_image = np.uint8(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX))
    
    return magnitude_image

def fourier_threshold_inverse(img, threshold_type, threshold_val):
    """
    Apply a specified Fourier threshold filter to the image and then compute the inverse Fourier Transform.

    Parameters:
    img (numpy.ndarray): The input image.
    threshold_type (str): The type of Fourier threshold filter to apply ('highPassFft', 'lowPassFft', 'gHighPassFft', 'gLowPassFft').
    threshold_val (int): The threshold value for the filter.

    Returns:
    tuple: The inverse Fourier transformed image and the log-transformed magnitude spectrum of the filtered Fourier transform.
    """
    gray_img = convert_to_grayscale(img)
    ft = calculate_2dft(gray_img)

    if threshold_type == 'highPassFft':
        fft_filtered = high_pass_filter_fft(ft, threshold_val)
    elif threshold_type == 'lowPassFft':
        fft_filtered = low_pass_filter_fft(ft, threshold_val)
    elif threshold_type == 'gHighPassFft':
        fft_filtered = gaussian_high_pass_filter_fft(ft, threshold_val)
    elif threshold_type == 'gLowPassFft':
        fft_filtered = gaussian_low_pass_filter_fft(ft, threshold_val)

    ifft_filtered = inverse_Fourier(fft_filtered)

    return ifft_filtered, 20 * np.log(np.abs(fft_filtered + 1))

def edge_detection(img, selected_edge_detection):
    """
    Perform edge detection on the given image using the specified method.

    Parameters:
    img (numpy.ndarray): The input image.
    selected_edge_detection (str): The edge detection method to use ('selectedEdgeSobel', 'selectedEdgeCanny', 'selectedEdgePrewitt', 'selectedEdgeRobertCross', 'selectedEdgeLaplacian', 'selectedEdgeScharr').

    Returns:
    numpy.ndarray: The image with edges detected.
    """
    if selected_edge_detection == "selectedEdgeSobel":
        image_8u = cv2.convertScaleAbs(img)
        gray_image = cv2.cvtColor(image_8u, cv2.COLOR_RGB2GRAY)
        edges = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    
    elif selected_edge_detection == "selectedEdgeCanny": 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=30, threshold2=150)
    
    elif selected_edge_detection == "selectedEdgePrewitt":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Vertical edges
        Gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # Horizontal edges
        edge_x = cv2.filter2D(gray, -1, Gx)
        edge_y = cv2.filter2D(gray, -1, Gy)
        edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
        edges = np.uint8(edge_magnitude / np.max(edge_magnitude) * 255)
    
    elif selected_edge_detection == "selectedEdgeRobertCross":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Gx = np.array([[1, 0], [0, -1]])
        Gy = np.array([[0, 1], [-1, 0]])
        edge_x = cv2.filter2D(gray, -1, Gx, borderType=cv2.BORDER_DEFAULT)
        edge_y = cv2.filter2D(gray, -1, Gy, borderType=cv2.BORDER_DEFAULT)
        edge_magnitude = np.sqrt(np.square(edge_x) + np.square(edge_y))
        edges = np.uint8(edge_magnitude / np.max(edge_magnitude) * 255)
    
    elif selected_edge_detection == "selectedEdgeLaplacian":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        edges = np.uint8(np.absolute(laplacian))
    
    elif selected_edge_detection == "selectedEdgeScharr":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scharrX = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharrY = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        scharrX = cv2.convertScaleAbs(scharrX)
        scharrY = cv2.convertScaleAbs(scharrY)
        edges = cv2.addWeighted(scharrX, 0.5, scharrY, 0.5, 0)

    return edges

def mean_shift_cluster(img):
    """
    Perform mean shift clustering on the given image and apply a colormap to the segmented result.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The segmented image with applied colormap.
    """
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    bandwidth = 30
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    meanshift.fit(pixel_values)
    labels = meanshift.labels_
    segmented_image = labels.reshape(img.shape[:2])
    normalized_labels = labels / labels.max()
    normalized_labels = normalized_labels.reshape(img.shape[:2])
    colored_image = plt.cm.jet(normalized_labels)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
    return colored_image

def k_means_cluster(img, num_centers):
    """
    Perform K-means clustering on the given image and apply a colormap to the segmented result.

    Parameters:
    img (numpy.ndarray): The input image.
    num_centers (int): The number of clusters (centers) for K-means clustering.

    Returns:
    numpy.ndarray: The segmented image with applied colormap.
    """
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = num_centers
    _, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    normalized_labels = labels / (num_centers - 1)
    normalized_labels = normalized_labels.reshape(img.shape[:2])
    colored_image = plt.cm.jet(normalized_labels)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
    return colored_image

def db_scan_cluster(img):
    """
    Perform DBSCAN clustering on the given image and assign random colors to different clusters.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The segmented image with clusters identified by random colors.
    """
    # Reshape and scale the image
    pixels = img.reshape((-1, 3))
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=10)
    clusters = dbscan.fit_predict(pixels_scaled)

    # Find unique labels
    unique_labels = np.unique(clusters)

    # Create an image to store the segmented result
    segmented_image = np.zeros(img.shape, dtype=np.uint8)

    # Assign random colors to different clusters
    for label in unique_labels:
        if label == -1:  # Noise
            color = [0, 0, 0]  # Black for noise
        else:
            color = np.random.randint(0, 255, size=3)
        mask = clusters == label
        segmented_image.reshape((-1, 3))[mask] = color

    return segmented_image

def img_cluster_segmentation(img, image_cluster_seg, num_centers):
    """
    Perform image clustering segmentation using the specified method.

    Parameters:
    img (numpy.ndarray): The input image.
    image_cluster_seg (str): The clustering method to use ('clusterKmeans', 'clusterMean', 'clusterDb').
    num_centers (int): The number of clusters (centers) for K-means clustering.

    Returns:
    numpy.ndarray: The segmented image.
    """
    if image_cluster_seg == "clusterKmeans":
        cluster_img = k_means_cluster(img, num_centers)
    elif image_cluster_seg == "clusterMean":
        cluster_img = mean_shift_cluster(img)
    elif image_cluster_seg == "clusterDb":
        cluster_img = db_scan_cluster(img)

    return cluster_img

def watershed_segmentation(img):
    """
    Perform watershed segmentation on the given image to separate foreground and background.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The image with boundaries marked using the watershed algorithm.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(img, markers)
    img[markers == -1] = [0, 255, 0]

    return img

def binary_class_pred(img, model_name):
    """
    Predict the binary class of the given image using the specified model.

    Parameters:
    img (numpy.ndarray): The input image.
    model_name (str): The name of the model to use for prediction ('customModelBin', 'vgg16Bin').

    Returns:
    float: The prediction score for the binary class.
    """
    target_size = (224, 224)
    resized_image = cv2.resize(img, target_size)
    img_array = img_to_array(resized_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    if model_name == 'customModelBin':
        model = load_model(custom_model_path)
    elif model_name == "vgg16Bin":
        model = load_model(vgg16_model_path)
    else:
        model = load_model(resnet_model_path)

    prediction = model.predict(img_array)

    return prediction[0][0]

def predict_yolo(chosen_model, img, classes=[], conf=0.5):
    """
    Perform object detection on the given image using the YOLO model.

    Parameters:
    chosen_model (keras.Model): The YOLO model to use for prediction.
    img (numpy.ndarray): The input image.
    classes (list): A list of class names to detect. If empty, all classes will be detected.
    conf (float): The confidence threshold for detection.

    Returns:
    list: The detection results.
    """
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect_yolo(chosen_model, img, classes=[], conf=0.5):
    """
    Perform object detection using YOLO model and draw bounding boxes with class labels on the image.

    Parameters:
    chosen_model (keras.Model): The YOLO model to use for prediction.
    img (numpy.ndarray): The input image.
    classes (list): A list of class names to detect. If empty, all classes will be detected.
    conf (float): The confidence threshold for detection.

    Returns:
    tuple: The image with bounding boxes and class labels drawn, and the detection results.
    """
    results = predict_yolo(chosen_model, img, classes, conf=conf)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results

def faster_rcnn_pred(img):
    """
    Perform object detection using Faster R-CNN model and draw bounding boxes with class labels on the image.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The image with bounding boxes and class labels drawn.
    """
    # Define the COCO object detection labels
    COCO_LABELS = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # Recreate the model architecture
    model = fasterrcnn_resnet50_fpn(pretrained=False)

    # Load the saved state dict
    model.load_state_dict(torch.load(fasterrccn_model_path))

    model.eval()  # Set it to evaluation mode

    # Convert PIL image to tensor
    img_tensor = F.to_tensor(img)

    # Make predictions
    with torch.no_grad():
        predictions = model([img_tensor])

    # Process predictions
    pred_boxes = predictions[0]['boxes'].detach().numpy()
    pred_labels = predictions[0]['labels'].detach().numpy()
    pred_scores = predictions[0]['scores'].detach().numpy()

    threshold = 0.5
    filtered_indices = np.where(pred_scores > threshold)
    filtered_boxes = pred_boxes[filtered_indices]

    # Draw boxes and labels on the image
    for i in filtered_indices[0]:
        box = pred_boxes[i].astype(int)
        label = COCO_LABELS[pred_labels[i]]
        score = pred_scores[i]

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{label}: {score:.2f}"
        cv2.putText(img, label_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img

def object_detection(img, detection_model):
    """
    Perform object detection on the given image using the specified detection model.

    Parameters:
    img (numpy.ndarray): The input image.
    detection_model (str): The detection model to use ('fasterRCnn', 'yolo').

    Returns:
    numpy.ndarray: The image with detected objects highlighted.
    """
    if detection_model == 'fasterRCnn':
        img = faster_rcnn_pred(img)
    elif detection_model == 'yolo':
        model = YOLO("yolov8n.pt")
        img, _ = predict_and_detect_yolo(model, img)
    return img

def xception_model(img):
    """
    Predict the top-3 classes of the given image using the pre-trained Xception model.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    list: A list of the top-3 predicted classes with their probabilities.
    """
    # Load the pre-trained Xception model
    model = Xception(weights='imagenet')

    # Convert the image to a numpy array and add a batch dimension
    target_size = (299, 299)
    img = cv2.resize(img, target_size)

    # Preprocess the image
    img_process = preprocess_input(np.expand_dims(img, axis=0))

    # Make predictions
    predictions = model.predict(img_process)

    # Decode top-3 predicted classes
    return decode_predictions(predictions, top=3)[0]

def inception_model(img):
    """
    Predict the top-3 classes of the given image using the pre-trained InceptionV3 model.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    list: A list of the top-3 predicted classes with their probabilities.
    """
    # Load the pre-trained InceptionV3 model
    model = InceptionV3(weights='imagenet')

    # Convert the image to a numpy array and add a batch dimension
    target_size = (299, 299)
    img = cv2.resize(img, target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    preds = model.predict(x)

    return decode_predictions(preds, top=3)[0]

def multiclass_clas(img, model):
    """
    Predict the top classes of the given image using the specified pre-trained model.

    Parameters:
    img (numpy.ndarray): The input image.
    model (str): The name of the model to use for prediction ('xceptionModel', 'inceptionV3').

    Returns:
    list: A list of the top predicted classes with their probabilities.
    """
    if model == 'xceptionModel':
        preds = xception_model(img)
    elif model == 'inceptionV3':
        preds = inception_model(img)
    return preds

def img_segmentation(img):
    """
    Perform image segmentation using YOLO model and return annotated image, class probabilities dataframe, and predictions.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    tuple: The annotated image, class probabilities dataframe, and predictions.
    """
    model = YOLO('yolov8n-seg.pt')
    preds = model.predict(img)[0]
    annotatedImageRGB = cv2.cvtColor(preds.plot(), cv2.COLOR_RGB2BGR)

    cat_lst = []
    pred_lst = []

    for i in range(len(preds)):
        cat_lst.append(preds.names[int(preds[i].boxes.data[0][-1])])
        pred_lst.append(round(float(preds[i].boxes.data[0][-2]), 2))

    df_class_prob = pd.DataFrame({
        'Classes': cat_lst,
        'Probabilities': pred_lst
    })
    
    df_class_prob['Row Num'] = range(1, len(df_class_prob) + 1)

    return annotatedImageRGB, df_class_prob, preds

def cumulative_division(segments):
    """
    Calculate cumulative divisions for a given number of segments.

    Parameters:
    segments (int): The number of segments.

    Returns:
    list: A list of cumulative divisions.
    """
    segment_size = 1 / segments
    cumulative_result = [segment_size * (i + 1) * 100 for i in range(segments)]
    return cumulative_result[::-1]

def thresh_clust(img, num_seg):
    """
    Perform threshold clustering on the given image and apply a colormap.

    Parameters:
    img (numpy.ndarray): The input image.
    num_seg (int): The number of segments.

    Returns:
    numpy.ndarray: The segmented image with applied colormap.
    """
    cut_offs = cumulative_division(num_seg)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arr = gray_image.flatten()

    for i in range(len(arr)):
        assigned_val = False
        for count, cut_off in enumerate(cut_offs):
            if arr[i] >= cut_off:
                arr[i] = count + 1
                assigned_val = True
                break
        if not assigned_val:
            arr[i] = 0
    gray_segmented = arr.reshape(gray_image.shape[0], gray_image.shape[1])
    colored_image = plt.cm.jet(gray_segmented / num_seg)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
    return colored_image

def image_seg_selection(results, user_choices):
    """
    Select image segments based on user choices and return annotated image and list of cropped images.

    Parameters:
    results (YOLOv8Result): The segmentation results from YOLOv8 model.
    user_choices (dict): The user choices for segmentation options.

    Returns:
    tuple: The annotated image and list of cropped images.
    """
    cropped_images_lst = []
    user_row_choice = [i - 1 for i in user_choices['rowNumbers']]
    annotatedImageRGB = cv2.cvtColor(results[user_row_choice].plot(masks=user_choices['options']['segMasksCheck'], boxes=user_choices['options']['segBbCheck']), cv2.COLOR_RGB2BGR)
    
    if user_choices['options']['segOutlinesCheck']:
        annotatedImageRGB_copy = Image.fromarray(cv2.cvtColor(annotatedImageRGB, cv2.COLOR_RGB2BGR))
        for i in user_row_choice:
            if results[i].masks:
                mask1 = results[i].masks[0]
                polygon = mask1.xy[0]
                draw = ImageDraw.Draw(annotatedImageRGB_copy)
                draw.polygon(polygon, outline=(0, 255, 0), width=5)
        annotatedImageRGB = np.array(annotatedImageRGB_copy)
        annotatedImageRGB = annotatedImageRGB[..., ::-1]
    
    if user_choices['options']['segCutCheck']:
        annotatedImageRGB_copy = np.array(Image.fromarray(cv2.cvtColor(annotatedImageRGB, cv2.COLOR_RGB2BGR)))
        for i in user_row_choice:
            b_mask = np.zeros(annotatedImageRGB_copy.shape[:2], np.uint8)
            contour = np.array(results[i].masks.xy).reshape(-1, 1, 2).astype(np.int32)
            trial_img = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            isolated_transparent = np.dstack([annotatedImageRGB_copy, b_mask])
            x1, y1, x2, y2 = results[i].boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            iso_crop = isolated_transparent[y1:y2, x1:x2]
            cropped_images_lst.append(iso_crop)

    return annotatedImageRGB, cropped_images_lst

def edit_image(image_path):
    """
    Simulate the process of editing an image.

    Parameters:
    image_path (str): The file path of the input image.

    Returns:
    str: The file path of the edited image.
    """
    img = imageio.imread(image_path)
    edited_image = img  # Your actual editing logic would modify this
    edited_path = image_path.replace('.jpg', '_edited.jpg')
    imageio.imwrite(edited_path, edited_image)
    return edited_path

def custom_seg_model(img):
    """
    Perform custom segmentation on the given image using a YOLO model.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    tuple: The annotated image and a message indicating whether detections were made.
    """
    nose_model = YOLO(best_model_path)
    results = nose_model.predict(img)[0]
    annotatedImageRGB = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)

    if results.boxes and len(results.boxes.xyxy) > 0:
        found_nose = "Detections were made."
    else:
        found_nose = "No detections."

    return annotatedImageRGB, found_nose

def img_to_text(file, file_type):
    """
    Extract text from an image or PDF file using OCR.

    Parameters:
    file (str): The file path of the input image or PDF.
    file_type (str): The type of the input file ('image/jpeg' or 'application/pdf').

    Returns:
    tuple: A list of images and a list of extracted text strings.
    """
    text_lst = []
    img_lst = []

    if file_type == 'image/jpeg':
        text = pytesseract.image_to_string(file)
        text_lst.append(text)
        img_lst.append(file)

    if file_type == 'application/pdf':
        images = convert_from_path(file, dpi=300)
        for image in images:
            text = pytesseract.image_to_string(image)
            text_lst.append(text)
            img_lst.append(image)

    return img_lst, text_lst

def chat_gpt(api_key, text, question):
    """
    Interact with OpenAI's GPT-3.5-turbo model using the specified text and question.

    Parameters:
    api_key (str): The API key for authenticating with OpenAI.
    text (str): The input text to be sent to the model.
    question (str): The question to be sent to the model.

    Returns:
    str: The response from the GPT-3.5-turbo model.
    """
    api_url = 'https://api.openai.com/v1/chat/completions'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'system', 'content': question},
            {'role': 'user', 'content': text},
        ]
    }
        
    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return 'Error: No response content available'
    else:
        return f'Error: {response.status_code} - {response.text}'


def predict_sign_language(base64_img):
    # Path to your pickled model
    model_path = 'models/asl_xgboost_model_21_aug.pkl'

    # Load the pickled XGBoost model
    with open(model_path, 'rb') as file:
        xgb_model_21_aug = pickle.load(file)

    # Load the LabelEncoder object from the file
    with open('models/label_encoder_asl.pkl', 'rb') as file:
        label_encoder = pickle.load(file)

    # Decode base64 image to a numpy array
    header, encoded = base64_img.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    rgb_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # Set up the MediaPipe Hands model
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Convert the frame to RGB as MediaPipe uses RGB images
    rgb_frame = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # Process the frame for hand tracking
    result = hands.process(rgb_frame)

    # If hands are detected, draw landmarks and connections and make predictions
    flattened_array = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the landmarks on the original frame
            mp_draw.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Flatten the x, y, z coordinates of the landmarks
            flattened_list = [value for landmark in hand_landmarks.landmark for value in (landmark.x, landmark.y, landmark.z)]
            flattened_array = np.array(flattened_list).reshape(1, -1)  # Reshape for the model input

    if flattened_array is not None:
        # Make prediction using the XGBoost model
        prediction_xgb = xgb_model_21_aug.predict_proba(flattened_array)

        # Convert the combined numeric prediction back to the letter using inverse_transform
        numeric_prediction = np.argmax(prediction_xgb)
        predicted_letter = label_encoder.inverse_transform([numeric_prediction])[0]  # Decode the predicted letter
 
        return predicted_letter
    else:
        return "No hand detected"







