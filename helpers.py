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
        cv2.drawContours(img,[box],0,(0,0,255),2)

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

    # Optionally, map the cluster labels to colors for visualization
    cluster_centers = meanshift.cluster_centers_
    segmented_image_color = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(len(cluster_centers)):
        segmented_image_color[segmented_image == i] = cluster_centers[i]


    return segmented_image_color

def k_means_cluster(img):

    # Reshape the image to a 2D array of pixels
    pixel_values = img.reshape((-1, 3))
    # Convert to float
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 3 # Number of clusters
    _, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Flatten the labels array
    labels = labels.flatten()

    # Convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)
    # Convert to BGR for displaying with OpenCV
    return segmented_image

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


def img_cluster_segmentation(img,image_cluster_seg):

    if image_cluster_seg == "clusterKmeans":
        cluster_img = k_means_cluster(img)
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
    
    if model_name == 'customModelBin':
        target_size=(150,150)
    else:
        target_size=(224, 224)

    resized_image = cv2.resize(img, target_size) 
    img_array = img_to_array(resized_image)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.  # Scale pixel values to [0, 1]

    if model_name == 'customModelBin':
        model = load_model(r'C:\Users\user\OneDrive\Desktop\trial_notebooks\custom.h5')
    elif model_name == "vgg16Bin":  
        model = load_model(r'C:\Users\user\OneDrive\Desktop\trial_notebooks\vgg16.h5')
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
    

def rcnn_pred(img,model):
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
    if detection_model == 'rCnn':
        img = rcnn_pred(img)
    
    if detection_model == 'yolo':
        model = YOLO("yolov8n.pt") 
        img, results = predict_and_detect_yolo(model,img)
    return img

    
     


