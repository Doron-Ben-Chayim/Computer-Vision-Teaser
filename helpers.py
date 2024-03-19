import cv2
import numpy as np
from math import sqrt,exp

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

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base

def butterworthLP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def butterworthHP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1-1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def fourrier(img):
    # Convert to grayscale
    gray_img = convert_to_grayscale(img)
    ft = calculate_2dft(gray_img)

    magnitude_spectrum = 20*np.log(np.abs(ft))

    # Convert the magnitude spectrum to a format that can be displayed
    # Normalize to fit [0, 255] and convert to uint8
    magnitude_image = np.uint8(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX))
    
    return magnitude_image

    #Show filtered image with filtered fourier Spectrum
    # Normal High Pass
    # Normal Low Pass
    # Gauss HP
    # Gauss LP
    # Butterworth HP
    # Butterworh LP
  


    
  











    
    
    
     


