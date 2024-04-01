from flask import Flask, render_template, request, jsonify, abort
import helpers as hlprs
import numpy as np
import pickle
import cv2
import base64

app = Flask(__name__)

@app.route('/')
def render_html():
    return render_template('index.html')

@app.route('/kernel_popup')
def kernel_popup():
    return render_template('kernel_popup.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data.get('imageData')
    image_height = data.get('imageHeight')
    image_width = data.get('imageWidth')
    image_process = data.get('imageProcess')
    translate_distances = data.get('imageTranslateDistances')
    image_width_selected = data.get('imageWidthSelected')
    image_height_selected = data.get('imageHeightSelected')
    image_rotate_angle = data.get('imageRotateAngle')
    image_colour_choice = data.get('imageColourChoice')
    image_current_colour_scheme = data.get('imageCurrentColourScheme')
    image_simple_threshold = data.get('imageselectedSimpleThreshold')
    image_threshold_value = data.get('imagethresholdValue')
    image_threshold_max = data.get('imagethresholdMax')
    image_affine_transform = data.get('imageAffineTransform')
    image_adaptive_paramaters = data.get('imageAdaptiveParamaters')
    image_selected_kernel = data.get('imageselectedKernel')
    image_morph_selection = data.get('imageMorphSelection')
    image_contour_feature_selection = data.get('imageContourFeatureSelection')
    image_contour_bounding_box_selection = data.get('imageContourBoundingBoxSelection')
    image_fft_Filter_Selection = data.get('imagefftFilterSelection')
    image_selected_edge_detection = data.get('imageSelectedEdgeDetection') 
    image_cluster_seg = data.get('imageClusterSeg')                  
                               

    # print(image_data)
    pixel_data = image_data['data']
    pixel_data = np.array(list(pixel_data.values()))

    red_array = pixel_data[2::4]
    green_array = pixel_data[1::4]
    blue_array = pixel_data[::4]

    rgb_image_array = np.column_stack((red_array, green_array, blue_array)).reshape(image_width,image_height,3).astype(np.uint8)

    # Save the image data as a pickle file
    pickle_file_path = 'image_data_before.pickle'   
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(rgb_image_array , file)
    
    histr = ''
    amplitude_threshold = ''
    
    # Process the image data in your Python script
    print(image_process)
    if image_process == 'resize':
        image_data_array_edited = hlprs.resize_image(rgb_image_array,image_width_selected,image_height_selected)
    if image_process == 'translate':
        print('TRANSLATE')
        image_data_array_edited = hlprs.translate_image(rgb_image_array,translate_distances)
    if image_process == 'affine':
        image_data_array_edited = hlprs.affine_transformation(rgb_image_array, image_affine_transform)
    if image_process == 'swapColour':
        image_data_array_edited = hlprs.swap_colour(rgb_image_array,image_colour_choice,image_current_colour_scheme)
    if image_process == 'crop':
        image_data_array_edited = rgb_image_array
    if image_process == 'rotate':
        image_data_array_edited = hlprs.rotate_image(rgb_image_array,image_rotate_angle)
    if image_process == 'grayscale':
        image_data_array_edited = hlprs.convert_to_grayscale(rgb_image_array)
    if image_process == 'smoothingKernel':
        image_data_array_edited = hlprs.smooth_kernel(rgb_image_array,image_selected_kernel)
    if image_process == 'edgeDetectionKernel':
        image_data_array_edited = hlprs.edge_kernel(rgb_image_array,image_selected_kernel)
    if image_process == 'simpleThresh':
        image_data_array_edited = hlprs.simple_thresh(rgb_image_array,image_simple_threshold,image_threshold_value,image_threshold_max)
    if image_process == 'adaptThresh':
        image_data_array_edited = hlprs.adapt_thresh(rgb_image_array,image_adaptive_paramaters)
    if image_process == 'otsuThresh':
        image_data_array_edited = hlprs.otsu_thresh(rgb_image_array,image_threshold_value,image_threshold_max)
    if image_process == 'imageHist':
        image_data_array_edited, histr = hlprs.get_hist(rgb_image_array)
        histr = [hist.flatten().tolist() for hist in histr]
    if image_process == 'histEqua':
        image_data_array_edited, histr = hlprs.hist_equalization(rgb_image_array)
        histr = [hist.flatten().tolist() for hist in histr]
    if image_process == 'customKernel':
        image_data_array_edited = hlprs.custom_kernel(rgb_image_array,image_selected_kernel)
    if image_process == 'morphologicalKernel':
        image_data_array_edited = hlprs.dilate_image(rgb_image_array,image_morph_selection)
    if image_process == 'drawContours':
        image_data_array_edited = hlprs.draw_contours(rgb_image_array)
    if image_process == 'contourFeatures':
        image_data_array_edited = hlprs.show_contour_properties(rgb_image_array,image_contour_feature_selection)
    if image_process == 'boundingFeatures':
        image_data_array_edited = hlprs.show_contour_bounding_box(rgb_image_array,image_contour_bounding_box_selection)
    if image_process == 'FftSpectrum':
        image_data_array_edited = hlprs.fourier_spectrum_20(rgb_image_array)
    if image_process == 'FftFilter':
        image_data_array_edited, amplitude_threshold = hlprs.fourier_threshold_inverse(rgb_image_array, image_fft_Filter_Selection,10) #fft_threshold
        # Convert the amplitude_threshold to a base64-encoded string
        _, buffer = cv2.imencode('.png', amplitude_threshold)
        amplitude_threshold = base64.b64encode(buffer).decode('utf-8')
    if image_process == 'edgeDetection':
        image_data_array_edited = hlprs.edge_detection(rgb_image_array,image_selected_edge_detection)
    if image_process == 'clusterSeg':
        image_data_array_edited = hlprs.img_cluster_segmentation(rgb_image_array,image_cluster_seg)
    if image_process == 'watershed':
        image_data_array_edited = hlprs.watershed_segmentation(rgb_image_array)
            
        
    # Specify the file path where you want to save the pickle file
    pickle_file_path = 'image_data_after.pickle'
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(image_data_array_edited, file)
    
    pickle_file_path = 'image_data_after_histr.pickle'
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(amplitude_threshold, file)

    # Convert the NumPy array to a base64-encoded string
    _, buffer = cv2.imencode('.png', image_data_array_edited)
    image_data = base64.b64encode(buffer).decode('utf-8')
    
    # Dummy response for demonstration purposes
    response = {'status': 'success', 'img': image_data, 'currentColourScheme':image_colour_choice, 'histogramVals': histr, "fftThresh": amplitude_threshold }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)


#http://127.0.0.1:5000
    
