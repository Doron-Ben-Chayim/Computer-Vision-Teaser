import cv2
import numpy as np
from math import sqrt,exp
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import torch
from IPython.display import display
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
import base64
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import imageio
import pytesseract
from pdf2image import convert_from_path


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
    scaling_factor = float(affine_params[1])
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

def resize_image(img,new_width, new_height):
    new_width = int(new_width)
    new_height = int(new_height)

    resized_image = cv2.resize(img, (new_width, new_height))
    return resized_image

import numpy as np
import cv2

def reconstruct_image(pixel_data, image_width, image_height, input_format, output_format):
    # Convert the flat list of values into a numpy array
    pixel_data = np.array(list(pixel_data.values()), dtype=np.uint8)

    # Initialize image_array
    image_array = None

    if input_format == 'hsvColour':
        input_format = 'rgbColour'

    # Extract channels based on the input format and reshape
    if input_format in ['rgbColour', 'bgrColour']:
        if input_format == 'rgbColour':
            channel_indices = [2, 1, 0]  # R, G, B
        elif input_format == 'bgrColour':
            channel_indices = [0, 1, 2]  # B, G, R

        red = pixel_data[channel_indices[0]::4]
        green = pixel_data[channel_indices[1]::4]
        blue = pixel_data[channel_indices[2]::4]
        image_array = np.stack((red, green, blue), axis=-1).reshape(image_height, image_width, 3)

    elif input_format == 'hsvColour':
        hue = pixel_data[0::3]
        saturation = pixel_data[1::3]
        value = pixel_data[2::3]
        image_array = np.stack((hue, saturation, value), axis=-1).reshape(image_height, image_width, 3)

        # Convert colour to BGR 
    if output_format == 'bgrColour':
        if input_format == 'bgrColour':
            colour_swapped_image = image_array
            print('1')
        elif input_format == 'hsvColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_HSV2BGR)
            print('2')
        elif input_format == 'rgbColour':
            
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            print('3')
    
    if output_format == 'hsvColour':
        if input_format == 'hsvColour':
            colour_swapped_image = image_array
            print('4')
        elif input_format == 'bgrColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
            print('5')
        elif input_format == 'rgbColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            print('6')
    
    if output_format == 'rgbColour':
        if input_format == 'rgbColour':
            colour_swapped_image = image_array
            print('7')
        elif input_format == 'bgrColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            print('8')
        elif input_format == 'hsvColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_HSV2RGB)
            print('9')

    return colour_swapped_image



def swap_colour(image_array,image_colour_choice,image_current_colour_scheme):
    print(image_colour_choice,"image_colour_choice")
    print(image_current_colour_scheme,"image_current_colour_scheme")
    
    # Convert colour to BGR 
    if image_colour_choice == 'bgrColour':
        if image_current_colour_scheme == 'bgrColour':
            colour_swapped_image = image_array
            print('1')
        elif image_current_colour_scheme == 'hsvColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_HSV2BGR)
            print('2')
        elif image_current_colour_scheme == 'rgbColour':
            
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            print('3')
    
    if image_colour_choice == 'hsvColour':
        if image_current_colour_scheme == 'hsvColour':
            colour_swapped_image = image_array
            print('4')
        elif image_current_colour_scheme == 'bgrColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
            print('5')
        elif image_current_colour_scheme == 'rgbColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            print('6')
    
    if image_colour_choice == 'rgbColour':
        if image_current_colour_scheme == 'rgbColour':
            colour_swapped_image = image_array
            print('7')
        elif image_current_colour_scheme == 'bgrColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            print('8')
        elif image_current_colour_scheme == 'hsvColour':
            colour_swapped_image = cv2.cvtColor(image_array, cv2.COLOR_HSV2RGB)
            print('9')

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

def otsu_thresh(img,image_threshold_value,image_threshold_max):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # image = img.astype("uint8")
    # blur = cv2.GaussianBlur(img,(5,5),0)
    _,otsu_thresh_image = cv2.threshold(gray_image,int(image_threshold_value),int(image_threshold_max),cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return otsu_thresh_image

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

def smooth_kernel(img,image_selected_kernel):

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

def edge_kernel(img,image_selected_kernel):
    image_8u = cv2.convertScaleAbs(img)
    gray_image = cv2.cvtColor(image_8u, cv2.COLOR_RGB2GRAY)

    if image_selected_kernel == 'sobelXKernel':
        edge_image = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    elif image_selected_kernel == 'sobelYKernel':
        edge_image = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    elif image_selected_kernel == 'prewittXKernel':
        # Define the PrewittX kernel
        prewitt_x_kernel = np.array([[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]])

        # Apply the PrewittX kernel using cv2.filter2D()
        edge_image = cv2.filter2D(gray_image, cv2.CV_64F, prewitt_x_kernel)
    
    elif image_selected_kernel == 'prewittYKernel':
        # Define the PrewittY kernel
        prewitt_y_kernel = np.array([[-1, -1, -1],
                                    [0, 0, 0],
                                    [1, 1, 1]])

        # Apply the PrewittY kernel using cv2.filter2D()
        edge_image = cv2.filter2D(gray_image, cv2.CV_64F, prewitt_y_kernel)

    return edge_image

def custom_kernel(img, provided_kernel):
    # convert kernel to numpy array
    kernal_array = np.array(provided_kernel, dtype=float)
    # check for if need to grayscale/threshold before
    new_image = cv2.filter2D(img, -1, kernal_array)

    return new_image

def dilate_image(img, morph_selection):
    kernel = np.ones((5, 5), np.uint8) 
    
    if morph_selection == "dilateKernel":
        morph_image = cv2.dilate(img, kernel, iterations=1)
    elif  morph_selection == "erodeKernel":
        morph_image = cv2.erode(img, kernel, iterations=1)
    elif  morph_selection == "openKernel":
        morph_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif  morph_selection == "closeKernel":
        morph_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return morph_image

# def draw_shape(img, shape_type):
#     if shape_type == 'rectangle':
#         cv2.rectangle(img, top_left_tuple, bottom_right_tuple, (0, 255, 0), 3)
#     elif shape_type == 'cirlce':
#         cv2.circle(img,circle_centre_tuple, radius, (0,0,255), -1)     

def get_contorus(img):
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
    cnts = get_contorus(img)
    # Draw all contours on the original image
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 3)

    return img

def draw_area(img, cnts):
    for i, contour in enumerate(cnts):
        area = cv2.contourArea(contour)

        # Draw contour on the image
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

        # Add text annotation with contour area
        cv2.putText(img, f"{i+1}: {round(area,2)}", (int(contour[0][0][0]), int(contour[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

def draw_perimeter(img, cnts):
    for i, contour in enumerate(cnts):
        perimeter = cv2.arcLength(contour, True)

        # Draw contour on the image
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

        # Add text annotation with contour area
        cv2.putText(img, f"{i+1}: {round(perimeter,2)}", (int(contour[0][0][0]), int(contour[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

def draw_centre(img, cnts):

    for i, contour in enumerate(cnts):
        perimeter = cv2.arcLength(contour, True)
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
    cnts = get_contorus(img)

    if selected_property == 'contourArea':
        img = draw_area(img, cnts)
    elif selected_property == 'contourPerimeter':
        img = draw_perimeter(img, cnts)
    elif selected_property == 'contourCentre':
        img = draw_centre(img, cnts)
    
    return img

def draw_bounding_rectangle(img,cnts):
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    return img

def draw_bounding_rot_rectangle(img,cnts):
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,255,0),2)

    return img 

def draw_bounding_circle(img,cnts):
    for cnt in cnts:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(img,center,radius,(0,255,0),2)
    
    return img


def draw_bounding_ellipse(img,cnts):

    for cnt in cnts:
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(img,ellipse,(0,255,0),2)

    return img


def show_contour_bounding_box(img,bb_selection):

    cnts = get_contorus(img)

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
         

### Fourier

def calculate_2dft(input):
    ft = np.fft.fft2(input)
    ft = np.fft.fftshift(ft)
    return ft

def high_pass_filter_fft(ft, cutoff_frequency):
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
    rows, cols = ft.shape
    crow, ccol = rows // 2, cols // 2  # Center
    mask = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            mask[i, j] = np.exp(-(distance**2) / (2 * (cutoff_frequency**2)))
            
    # Apply the Gaussian low-pass filter
    fft_filtered = ft * mask
    return fft_filtered

def fourrier(img):
    # Convert to grayscale
    gray_img = convert_to_grayscale(img)
    ft = calculate_2dft(gray_img)

    return ft

def inverse_Fourier(ft):
    ift = np.fft.ifftshift(ft)
    ift = np.fft.ifft2(ift)  
    ift = ift.real  
    return ift

def fourier_spectrum_20(img):
    # Convert to grayscale
    gray_img = convert_to_grayscale(img)
    ft = calculate_2dft(gray_img)

    magnitude_spectrum = 20*np.log(np.abs(ft))

    # Convert the magnitude spectrum to a format that can be displayed
    # Normalize to fit [0, 255] and convert to uint8
    magnitude_image = np.uint8(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX))
    
    return magnitude_image


def fourier_threshold_inverse(img,threshold_type,threshold_val):

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

    ifft_filtered =  inverse_Fourier(fft_filtered)

    return ifft_filtered, 20*np.log(np.abs(fft_filtered+ 1)) 

 
def edge_detection(img,selected_edge_detection):

    if selected_edge_detection == "selectedEdgeSobel":
        image_8u = cv2.convertScaleAbs(img)
        gray_image = cv2.cvtColor(image_8u, cv2.COLOR_RGB2GRAY)
        edges = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    
    elif selected_edge_detection == "selectedEdgeCanny": 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=30, threshold2=150) 
    
    elif selected_edge_detection == "selectedEdgePrewitt":
        # Load the image in grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Define the Prewitt kernels for horizontal and vertical edge detection
        Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Vertical edges
        Gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # Horizontal edges
        
        # Apply the kernels to the image
        edge_x = cv2.filter2D(gray, -1, Gx)
        edge_y = cv2.filter2D(gray, -1, Gy)
        
        # Calculate the edge magnitude
        edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
        edges = np.uint8(edge_magnitude / np.max(edge_magnitude) * 255)  # Normalize to 0-255
    
    elif selected_edge_detection == "selectedEdgeRobertCross":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Define the Roberts Cross kernels
        Gx = np.array([[1, 0], [0, -1]])
        Gy = np.array([[0, 1], [-1, 0]])
        
        # Apply the kernels to the image
        # Note: We use 'valid' mode to avoid padding issues, resulting in a slightly smaller output image
        edge_x = cv2.filter2D(gray, -1, Gx, borderType=cv2.BORDER_DEFAULT)
        edge_y = cv2.filter2D(gray, -1, Gy, borderType=cv2.BORDER_DEFAULT)
        
        # Calculate the edge magnitude
        edge_magnitude = np.sqrt(np.square(edge_x) + np.square(edge_y))
        edges = np.uint8(edge_magnitude / np.max(edge_magnitude) * 255)
    
    elif selected_edge_detection == "selectedEdgeLaplacian":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        edges = np.uint8(np.absolute(laplacian))
    elif selected_edge_detection == "selectedEdgeScharr":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply the Scharr operator
        scharrX = cv2.Scharr(gray, cv2.CV_64F, 1, 0)  # Gradient in the X direction
        scharrY = cv2.Scharr(gray, cv2.CV_64F, 0, 1)  # Gradient in the Y direction
        
        # Convert the gradients to absolute values
        scharrX = cv2.convertScaleAbs(scharrX)
        scharrY = cv2.convertScaleAbs(scharrY)
        
        # Combine the two gradients
        edges = cv2.addWeighted(scharrX, 0.5, scharrY, 0.5, 0) 

    return edges

def mean_shift_cluster(img):
    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = img.reshape((-1, 3))

    # Convert to float
    pixel_values = np.float32(pixel_values)

    # Define the bandwidth. This could also be estimated using estimate_bandwidth from sklearn
    # bandwidth = estimate_bandwidth(pixel_values, quantile=0.1, n_samples=100)
    bandwidth = 30

    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # Perform meanshift clustering
    meanshift.fit(pixel_values)
    labels = meanshift.labels_

    # Reshape the labels back to the original image shape
    segmented_image = labels.reshape(img.shape[:2])

    # Normalize the labels to range [0, 1] for applying colormap
    normalized_labels = labels / labels.max()
    normalized_labels = normalized_labels.reshape(img.shape[:2])

    # Apply the colormap
    colored_image = plt.cm.jet(normalized_labels)  # Apply colormap
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # Convert to 8-bit format

    return colored_image

def k_means_cluster(img,num_centers):

    # Reshape the image to a 2D array of pixels
    pixel_values = img.reshape((-1, 3))
    # Convert to float
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = num_centers # Number of clusters
    _, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Flatten the labels array
    labels = labels.flatten()

    # Convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)

    # Normalize the labels to range [0, 1] for applying colormap
    normalized_labels = labels / (num_centers - 1)
    normalized_labels = normalized_labels.reshape(img.shape[:2])

    # Apply the colormap
    colored_image = plt.cm.jet(normalized_labels)  # Apply colormap
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # Convert to 8-bit format

    return colored_image

def db_scan_cluster(img):

    # Reshape and scale the image
    # Reshape the image to a 2D array of pixels (ignoring color channel information for now)
    pixels = img.reshape((-1, 3))

    # Scale the pixel values to bring them into a similar range
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels)

    # Apply DBSCAN
    # The eps and min_samples parameters are crucial and affect the clustering result significantly
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


def img_cluster_segmentation(img,image_cluster_seg,num_centers):

    if image_cluster_seg == "clusterKmeans":
        cluster_img = k_means_cluster(img,num_centers)
    if image_cluster_seg == "clusterMean":
        cluster_img = mean_shift_cluster(img)
    if image_cluster_seg == "clusterDb":  
        cluster_img = db_scan_cluster(img)

    return cluster_img

def watershed_segmentation(img):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal (optional, helps in some cases)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area using distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Mark the region of unknown with zero
    markers[unknown==255] = 0

    # Apply the watershed algorithm
    cv2.watershed(img, markers)
    img[markers == -1] = [0,255,0] # Mark boundaries with green color

    return img

def binary_class_pred(img,model_name):
    
    target_size=(224, 224)

    resized_image = cv2.resize(img, target_size) 
    img_array = img_to_array(resized_image)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.  # Scale pixel values to [0, 1]

    if model_name == 'customModelBin':
        model = load_model(r'C:\Users\user\OneDrive\Desktop\trial_notebooks\custom.h5')
    elif model_name == "vgg16Bin":  
        print('BEFORE')
        model = load_model(r"C:\Users\user\OneDrive\Desktop\trial_notebooks\vgg16.h5")
        print('HELLO MADE IT')
    else:
        model = load_model(r'C:\Users\user\OneDrive\Desktop\trial_notebooks\resnet.h5')

    prediction = model.predict(img_array)

    return prediction[0][0]


def predict_yolo(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect_yolo(chosen_model, img, classes=[], conf=0.5):
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
    # Define the COCO object detection labels (assuming COCO labels for the pre-trained model)
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
    model_path = "fasterrcnn_resnet50_fpn.pth"
    model.load_state_dict(torch.load(model_path))

    model.eval()  # Set it to evaluation mode if you're making predictions

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
        # Prepare label text with confidence score
        label_text = f"{label}: {score:.2f}"
        
        # Put label text above the bounding box
        cv2.putText(img, label_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img

def object_detection(img,detection_model):
    if detection_model == 'fasterRCnn':
        img = faster_rcnn_pred(img)
    
    if detection_model == 'yolo':
        model = YOLO("yolov8n.pt") 
        img, _ = predict_and_detect_yolo(model,img)
    return img


def xception_model(img):

    # Load the pre-trained Xception model
    model = Xception(weights='imagenet')

    # Convert the image to a numpy array and add a batch dimension
    target_size = (299, 299)
    img = cv2.resize(img, target_size)

    # Preprocess the image
    img_process = preprocess_input(np.expand_dims(img, axis=0))

    # Make predictions
    predictions = model.predict(img_process)

    # Decode and print the top-3 predicted classes
    print(decode_predictions(predictions, top=3)[0])
    return decode_predictions(predictions, top=3)[0]

def inception_model(img):
    # Load the pre-trained InceptionV3 model
    print('Running Inception')
    model = InceptionV3(weights='imagenet')
    # Convert the image to a numpy array and add a batch dimension
    target_size = (299, 299)
    img = cv2.resize(img, target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]


def multiclass_clas(img,model):
    if model == 'xceptionModel':
        preds = xception_model(img)
    
    if model == 'inceptionV3':
        preds = inception_model(img)
    
    return preds

def img_segmentation(img):
    model = YOLO('yolov8n-seg.pt')
    preds = model.predict(img)[0]
    annotatedImageRGB = cv2.cvtColor(preds.plot(), cv2.COLOR_RGB2BGR)

    cat_lst = []
    pred_lst = []

    for i in range(len(preds)):
        cat_lst.append(preds.names[int(preds[i].boxes.data[0][-1])])
        pred_lst.append(round(float(preds[i].boxes.data[0][-2]),2))

    # Correct DataFrame creation:
    df_class_prob = pd.DataFrame({
        'Classes': cat_lst,
        'Probabilities': pred_lst
    })
    
    df_class_prob['Row Num'] = range(1, len(df_class_prob) + 1)

    return annotatedImageRGB,df_class_prob, preds

def cumulative_division(segments):
    # Each segment will be 1 divided by the number of segments
    segment_size = 1 / segments
    
    # Generate a list where each element is the cumulative sum up to that point
    cumulative_result = [segment_size * (i + 1) * 100 for i in range(segments)]
    
    # Return the list reversed
    return cumulative_result[::-1]

# Example usage
def thresh_clust(img, num_seg):
    cut_offs = cumulative_division(num_seg)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arr=gray_image.flatten()

    for i in range(len(arr)):
        assigned_val = False
        for count, cut_off in enumerate(cut_offs):
            if arr[i] >=   cut_off:
                arr[i] = count +1
                assigned_val = True
                break
        if assigned_val !=True:
            arr[i] = 0
    gray_segmented=arr.reshape(gray_image.shape[0],gray_image.shape[1])

    # Apply a colormap
    colored_image = plt.cm.jet(gray_segmented / num_seg)  # Normalize and apply colormap

    # Convert to 8-bit unsigned integer format
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

    return colored_image

def image_seg_selection(results,user_choices):
    cropped_images_lst = []
    user_row_choice = [i-1 for i in user_choices['rowNumbers']]
    annotatedImageRGB = cv2.cvtColor(results[user_row_choice].plot(masks=user_choices['options']['segMasksCheck'], boxes=user_choices['options']['segBbCheck']), cv2.COLOR_RGB2BGR)
    
    if user_choices['options']['segOutlinesCheck']:
        # Assuming 'img' is your original image loaded earlier in the code
        annotatedImageRGB_copy = Image.fromarray(cv2.cvtColor(annotatedImageRGB, cv2.COLOR_RGB2BGR))

        # Get the outline of the shape drawn over it
        for i in user_row_choice:
            if results[i].masks:  # Check if masks are available in the result
                mask1 = results[i].masks[0]  # Get the first mask
                polygon = mask1.xy[0]  # Assume 'xy' is a valid attribute containing polygon coordinates

                draw = ImageDraw.Draw(annotatedImageRGB_copy)
                draw.polygon(polygon, outline=(0, 255, 0), width=5)

        annotatedImageRGB = np.array(annotatedImageRGB_copy)
        annotatedImageRGB = annotatedImageRGB[..., ::-1]
    
    
    if user_choices['options']['segCutCheck']:
        
        annotatedImageRGB_copy = np.array(Image.fromarray(cv2.cvtColor(annotatedImageRGB, cv2.COLOR_RGB2BGR)))
        for i in user_row_choice:
            b_mask = np.zeros(annotatedImageRGB_copy.shape[:2], np.uint8)
            contour = np.array(results[i].masks.xy).reshape(-1, 1, 2).astype(np.int32)
            trial_img = cv2.drawContours(b_mask,
                                [contour],
                                -1,
                                (255, 255, 255),
                                cv2.FILLED)

            # Create 3-channel mask
            # mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)

            # Isolate object with binary mask
            # isolated_black = cv2.bitwise_and(mask3ch, img)
            isolated_transparent = np.dstack([annotatedImageRGB_copy, b_mask])

            x1, y1, x2, y2 = results[i].boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            # Crop image to object region
            iso_crop = isolated_transparent[y1:y2, x1:x2]
            cropped_images_lst.append(iso_crop)
            # plt.imshow(iso_crop)
            # plt.savefig(f'isolated_image_{i}.png', bbox_inches='tight', pad_inches=0)
            # plt.close()

    return annotatedImageRGB,cropped_images_lst

def edit_image(image_path):
    # Simulated image editing process
    img = imageio.imread(image_path)
    edited_image = img  # Your actual editing logic would modify this
    edited_path = image_path.replace('.jpg', '_edited.jpg')
    imageio.imwrite(edited_path, edited_image)
    return edited_path

def custom_seg_model(img):
    nose_model = YOLO(r"C:\Users\user\OneDrive\Desktop\trial_notebooks\custom_segmentation\runs\segment\train\weights\best.pt")
    plt.imshow(img)
    plt.show()
    results = nose_model.predict(img)[0]
    annotatedImageRGB = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)

    # Check if any detections were made
    if results.boxes and len(results.boxes.xyxy) > 0:
        found_nose = "Detections were made."
    else:
        found_nose = "No detections."

    return annotatedImageRGB, found_nose

def img_to_text(file, file_type):
    text_lst = []
    img_lst = []
    if file_type == 'image/jpeg':
        text = pytesseract.image_to_string(file)
        text_lst.append(text)
        img_lst.append(file)

    if file_type == 'application/pdf':
        # Convert PDF to images with a specific DPI
        images = convert_from_path(file, dpi=300)
        
        # Display images in the notebook
        for i, image in enumerate(images):

            text = pytesseract.image_to_string(image)
            text_lst.append(text)
            img_lst.append(image)

    return img_lst, text_lst


import requests

def chat_gpt(api_key, text, question):
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

        


