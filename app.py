from flask import Flask, render_template, request, jsonify, abort,  send_file, send_from_directory
import helpers as hlprs
import numpy as np
import cv2
import base64
import zipfile
from io import BytesIO
from tempfile import gettempdir
import os
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib.pyplot as plt
import time
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='flask_app.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')


# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Ensure this folder exists
ALLOWED_EXTENSIONS = {'pdf', 'jpeg', 'jpg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

df_img_seg = None
seg_model_results = None

predict_lambda_dict = {
    "classImageUpload": {
        'binaryClass': lambda rgb_image_array, bin_model_name: float(hlprs.binary_class_pred(rgb_image_array, bin_model_name)),
        'multiClass': lambda rgb_image_array, multi_model_name: [(id, label, float(prob)) for id, label, prob in hlprs.multiclass_clas(rgb_image_array, multi_model_name)],
        'objectDetection': lambda rgb_image_array, detection_model: (hlprs.object_detection(rgb_image_array, detection_model), True)
    }
}

process_lambda_dict = {
    'resize': lambda rgb_image_array, **kwargs: hlprs.resize_image(rgb_image_array, kwargs['image_width_selected'], kwargs['image_height_selected']),
    'translate': lambda rgb_image_array, **kwargs: hlprs.translate_image(rgb_image_array, kwargs['translate_distances']),
    'affine': lambda rgb_image_array, **kwargs: hlprs.affine_transformation(rgb_image_array, kwargs['image_affine_transform']),
    'swapColour': lambda rgb_image_array, **kwargs: hlprs.reconstruct_image(kwargs['image_data'], kwargs['image_width'], kwargs['image_height'], kwargs['image_current_colour_scheme'], kwargs['image_desired_color_choice']),
    'crop': lambda rgb_image_array, **kwargs: rgb_image_array,
    'rotate': lambda rgb_image_array, **kwargs: hlprs.rotate_image(rgb_image_array, kwargs['image_rotate_angle']),
    'grayscale': lambda rgb_image_array, **kwargs: hlprs.convert_to_grayscale(rgb_image_array),
    'smoothingKernel': lambda rgb_image_array, **kwargs: hlprs.smooth_kernel(rgb_image_array, kwargs['image_selected_kernel']),
    'edgeDetectionKernel': lambda rgb_image_array, **kwargs: hlprs.edge_kernel(rgb_image_array, kwargs['image_selected_kernel']),
    'sharpeningKernel': lambda rgb_image_array, **kwargs: hlprs.sharp_kernel(rgb_image_array, kwargs['image_selected_kernel']),
    'simpleThresh': lambda rgb_image_array, **kwargs: hlprs.simple_thresh(rgb_image_array, kwargs['image_simple_threshold'], kwargs['image_threshold_value'], kwargs['image_threshold_max']),
    'adaptThresh': lambda rgb_image_array, **kwargs: hlprs.adapt_thresh(rgb_image_array, kwargs['image_adaptive_paramaters']),
    'otsuThresh': lambda rgb_image_array, **kwargs: hlprs.otsu_thresh(rgb_image_array, kwargs['image_threshold_value'], kwargs['image_threshold_max']),
    'imageHist': lambda rgb_image_array, **kwargs: hlprs.get_hist(rgb_image_array),
    'histEqua': lambda rgb_image_array, **kwargs: hlprs.hist_equalization(rgb_image_array),
    'customKernel': lambda rgb_image_array, **kwargs: hlprs.custom_kernel(rgb_image_array, kwargs['image_selected_kernel']),
    'morphologicalKernel': lambda rgb_image_array, **kwargs: hlprs.dilate_image(rgb_image_array, kwargs['image_morph_selection']),
    'drawContours': lambda rgb_image_array, **kwargs: hlprs.draw_contours(rgb_image_array),
    'contourFeatures': lambda rgb_image_array, **kwargs: hlprs.show_contour_properties(rgb_image_array, kwargs['image_contour_feature_selection']),
    'boundingFeatures': lambda rgb_image_array, **kwargs: hlprs.show_contour_bounding_box(rgb_image_array, kwargs['image_contour_bounding_box_selection']),
    'identifyShapes': lambda rgb_image_array, **kwargs: hlprs.identify_shapes(rgb_image_array),
    'FftSpectrum': lambda rgb_image_array, **kwargs: hlprs.fourier_spectrum_20(rgb_image_array),
    'FftFilter': lambda rgb_image_array, **kwargs: hlprs.fourier_threshold_inverse(rgb_image_array, kwargs['image_fft_Filter_Selection'], int(kwargs['image_slider_output'])),
    'edgeDetection': lambda rgb_image_array, **kwargs: hlprs.edge_detection(rgb_image_array, kwargs['image_selected_edge_detection']),
    'threshSeg': lambda rgb_image_array, **kwargs: hlprs.thresh_clust(rgb_image_array, int(kwargs['image_slider_output'])),
    'clusterSeg': lambda rgb_image_array, **kwargs: hlprs.img_cluster_segmentation(rgb_image_array, kwargs['image_cluster_seg'], int(kwargs['image_slider_output'])),
    'watershed': lambda rgb_image_array, **kwargs: hlprs.watershed_segmentation(rgb_image_array),
    'semantic': lambda rgb_image_array, **kwargs: hlprs.img_segmentation(rgb_image_array)
}

@app.route('/')
def render_html():
    return render_template('mainIndex.html')

@app.route('/readme')
def readme():
    return render_template('ReadMe.html')

@app.route('/kernel_popup')
def kernel_popup():
    return render_template('kernel_popup.html')

@app.route('/imgSegTable')
def data():
    return jsonify(df_img_seg.to_dict(orient='records'))

@app.route('/download-edited-images')
def download_edited_images():
    return send_file('path/to/save/edited_images.zip', attachment_filename='edited_images.zip', as_attachment=True)

@app.route('/ask-chatgpt', methods=['POST'])
def askChatGPT():
    data = request.get_json()

    api_key = data.get('chatAPI')
    text = data.get('text')
    question = data.get('question')
 
    chat_response = hlprs.chat_gpt(api_key,text,question)

    response = {
        'status': 'success',
        'chatGPTResponse': chat_response,
    }

    return jsonify(response)

@app.route('/processSeg', methods=['POST'])
def process():
    data = request.get_json()
    print("Data received for processing:", data)
    
    # Assuming this function returns an edited main image and a list of cropped images
    image_data_array_edited, cropped_images_lst = hlprs.image_seg_selection(seg_model_results, data)
    image_data_array_edited = image_data_array_edited[..., ::-1]  # Convert BGR to RGB

    # Convert the main image to a base64-encoded string if it exists
    image_data = None
    if image_data_array_edited is not None:
        _, buffer = cv2.imencode('.png', image_data_array_edited)
        image_data = base64.b64encode(buffer).decode('utf-8')

    # Prepare cropped images and save to a temporary zip if there are any
    file_url = None
    if cropped_images_lst:
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for idx, cropped_image in enumerate(cropped_images_lst):
                _, buffer = cv2.imencode('.png', cropped_image)
                zf.writestr(f'cropped_image_{idx + 1}.png', buffer)
        
        memory_file.seek(0)
        # Save to a temporary file
        zip_filename = "processed_images.zip"
        zip_path = os.path.join(gettempdir(), zip_filename)  # Save in the system temporary directory
        with open(zip_path, 'wb') as f:
            f.write(memory_file.getvalue())
        file_url = '/download-zip'

    response = {
        'status': 'success',
        'img': image_data,
        'zip_url': file_url
    }
    
    return jsonify(response)

# Endpoint to download the zip file
@app.route('/download-zip')
def download_zip():
    zip_path = os.path.join(gettempdir(), "processed_images.zip")
    return send_file(zip_path, download_name='processed_images.zip', as_attachment=True)


@app.route('/predict-pdf', methods=['POST'])
def predict_OCR():
    print('ANALYSING PDF')
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    processed_image_lst, txt_lst  = hlprs.img_to_text(file_path,'application/pdf')
    
    processed_image_lst_converted = []

    for ind_img in processed_image_lst:
        
        # Save the PIL image to a bytes buffer in PPM format
        buffer = BytesIO()
        ind_img.save(buffer, format='JPEG')
        buffer.seek(0)  # Ensure the buffer's position is at the start
        
        # Encode the bytes buffer to base64
        processed_image = base64.b64encode(buffer.read()).decode('utf-8')
        processed_image_lst_converted.append(processed_image)

    response = {'status': 'success','imgsLst':processed_image_lst_converted, 'imgTxtsLst':txt_lst}
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict_img():
    
    data = request.get_json()
    image_data = data.get('predImageData')
    image_height = data.get('predImageHeight')
    image_width = data.get('predImageWidth')
    image_process = data.get('imageProcess')
    bin_model_name = data.get('binModel')
    multi_model_name = data.get('multiModel')
    detection_model = data.get('detectionModel')
    selected_task = data.get('selectedTask')
    file_type = data.get('fileType')
    
    # Process image data
    pixel_data = np.array(list(image_data.values()))

    red_array = pixel_data[2::4]
    green_array = pixel_data[1::4]
    blue_array = pixel_data[0::4]

    # Combine the R, G, and B arrays into a 3-channel 2D array
    rgb_image_array = np.stack((red_array, green_array, blue_array), axis=-1).reshape(image_height, image_width, 3).astype(np.uint8) 

    is_proccesed_image = False
    bin_pred_converted = False
    multi_pred = False
    found_nose = ''
    img_text_lst = []
    processed_image = ''
    processed_image_lst_converted = ''
    
    # Use the lambda dictionary for classImageUpload tasks
    if selected_task == "classImageUpload":
        result = predict_lambda_dict[selected_task][image_process](rgb_image_array, bin_model_name if image_process == 'binaryClass' else multi_model_name if image_process == 'multiClass' else detection_model)
        if image_process == 'binaryClass':
            bin_pred_converted = result
        elif image_process == 'multiClass':
            multi_pred = result
        elif image_process == 'objectDetection':
            processed_image, is_proccesed_image = result

    elif selected_task == 'segImageUpload':
        processed_image, found_nose = hlprs.custom_seg_model(rgb_image_array)
        processed_image = processed_image[..., ::-1]
        is_proccesed_image = True

    elif selected_task == 'ocrImageUpload':
        print('ANALYSING OCR IMAGE')
        processed_image_lst, img_text_lst  = hlprs.img_to_text(rgb_image_array, 'image/jpeg')
        processed_image_lst_converted = []
        _, buffer = cv2.imencode('.png', processed_image_lst[0])
        processed_image = base64.b64encode(buffer).decode('utf-8')
        processed_image_lst_converted.append(processed_image)
        print('PROCESSED THE IMAGE')
        is_proccesed_image = False

    # Convert the processed image or original image to base64
    if is_proccesed_image: 
        _, buffer = cv2.imencode('.png', processed_image)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        print('PROCESSED THE IMAGE')
    else:
        _, buffer = cv2.imencode('.png', rgb_image_array)
        processed_image = base64.b64encode(buffer).decode('utf-8')
  
    # Create the response
    response = {
        'status': 'success',
        'img': processed_image,
        'binPred': bin_pred_converted,
        'multiPred': multi_pred,
        "foundNose": found_nose,
        'processed_image_lst': processed_image_lst_converted,
        'imgTextOCR': img_text_lst
    }

    print('multi_pred', multi_pred)
    print('bin_pred_converted', bin_pred_converted)
    return jsonify(response)




@app.route('/process_image', methods=['POST'])
def process_image():
    start_request_time = time.time()
    data = request.get_json()
    request_sent_time = data.get('requestStartTime')
    image_data = data.get('imageData')
    image_height = data.get('imageHeight')
    image_width = data.get('imageWidth')
    image_process = data.get('imageProcess')
    translate_distances = data.get('imageTranslateDistances')
    image_width_selected = data.get('imageWidthSelected')
    image_height_selected = data.get('imageHeightSelected')
    image_rotate_angle = data.get('imageRotateAngle')
    image_desired_color_choice = data.get('imageDesiredColorScheme')
    image_current_colour_scheme = data.get('imageCurrentColourSchemeMain')
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
    image_slider_output = data.get('imageSliderOutput')

    end_request_time = time.time() - start_request_time
    app.logger.info(f"Request processing time: {end_request_time} seconds")

    if request_sent_time:
        request_received_time = start_request_time * 1000  # Convert to milliseconds to match the JavaScript timestamp
        transit_time = request_received_time - request_sent_time
        app.logger.info(f"Request transit time: {transit_time} ms")

    if image_process == 'identityKernel':
        return jsonify({'status': 'no action'})

    rgb_process_time_start = time.time()
    # Process image data
    pixel_data = np.array(list(image_data.values()))
    
    red_array = pixel_data[2::4]
    green_array = pixel_data[1::4]
    blue_array = pixel_data[0::4]

    # Combine the R, G, and B arrays into a 3-channel 2D array
    rgb_image_array = np.stack((red_array, green_array, blue_array), axis=-1).reshape(image_height, image_width, 3).astype(np.uint8)
    rgb_process_time_end = time.time() -  rgb_process_time_start
    app.logger.info(f"rgb processing time: {rgb_process_time_end} seconds")

    histr = ''
    amplitude_threshold = ''
    semantic_img = False
    
    # Process the image data using the lambda dictionary
    kwargs = {
        'image_data': image_data,
        'image_width': image_width,
        'image_height': image_height,
        'translate_distances': translate_distances,
        'image_width_selected': image_width_selected,
        'image_height_selected': image_height_selected,
        'image_rotate_angle': image_rotate_angle,
        'image_desired_color_choice': image_desired_color_choice,
        'image_current_colour_scheme': image_current_colour_scheme,
        'image_simple_threshold': image_simple_threshold,
        'image_threshold_value': image_threshold_value,
        'image_threshold_max': image_threshold_max,
        'image_affine_transform': image_affine_transform,
        'image_adaptive_paramaters': image_adaptive_paramaters,
        'image_selected_kernel': image_selected_kernel,
        'image_morph_selection': image_morph_selection,
        'image_contour_feature_selection': image_contour_feature_selection,
        'image_contour_bounding_box_selection': image_contour_bounding_box_selection,
        'image_fft_Filter_Selection': image_fft_Filter_Selection,
        'image_selected_edge_detection': image_selected_edge_detection,
        'image_cluster_seg': image_cluster_seg,
        'image_slider_output': image_slider_output
    }
    start_selected_process = time.time()
    if image_process in process_lambda_dict:
        if image_process == 'imageHist' or image_process == 'histEqua':
            image_data_array_edited, histr = process_lambda_dict[image_process](rgb_image_array, **kwargs)
            histr = [hist.flatten().tolist() for hist in histr]
        elif image_process == 'FftFilter':
            image_data_array_edited, amplitude_threshold = process_lambda_dict[image_process](rgb_image_array, **kwargs)
            _, buffer = cv2.imencode('.png', amplitude_threshold)
            amplitude_threshold = base64.b64encode(buffer).decode('utf-8')
        elif image_process == 'semantic':
            global df_img_seg, seg_model_results
            image_data_array_edited, df_img_seg, seg_model_results = process_lambda_dict[image_process](rgb_image_array, **kwargs)
            image_data_array_edited = image_data_array_edited[..., ::-1]
            semantic_img = True
        else:
            image_data_array_edited = process_lambda_dict[image_process](rgb_image_array, **kwargs)

    # Convert the NumPy array to a base64-encoded string
    _, buffer = cv2.imencode('.png', image_data_array_edited)
    image_data = base64.b64encode(buffer).decode('utf-8')

    end_selected_process = time.time() - start_selected_process  # End timing for grayscale processing
    app.logger.info(f"Total processing time: {end_selected_process} seconds")

    # Dummy response for demonstration purposes
    response = {
        'status': 'success',
        'img': image_data,
        'desiredColourScheme': image_desired_color_choice,
        'histogramVals': histr,
        "fftThresh": amplitude_threshold,
        'semanticBool': semantic_img
    }
    return jsonify(response)


if __name__ == '__main__':  
    
    app.run(debug=True)


#http://127.0.0.1:5000
    
