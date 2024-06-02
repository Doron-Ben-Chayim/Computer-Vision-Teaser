from flask import Flask, render_template, request, jsonify, abort,  send_file, send_from_directory
import helpers as hlprs
import numpy as np
import pickle
import cv2
import base64
import zipfile
from io import BytesIO
from tempfile import gettempdir
import os
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)


# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Ensure this folder exists
ALLOWED_EXTENSIONS = {'pdf', 'jpeg', 'jpg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

df_img_seg = None
seg_model_results = None

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
    
    # print(image_data)
    pixel_data = image_data
    pixel_data = np.array(list(pixel_data.values()))

    red_array = pixel_data[2::4]
    green_array = pixel_data[1::4]
    blue_array = pixel_data[0::4]

    # Combine the R, G, and B arrays into a 3-channel 2D array
    rgb_image_array = np.stack((red_array, green_array, blue_array), axis=-1).reshape(image_height, image_width, 3).astype(np.uint8) 

    # plt.imshow(rgb_image_array)
    # plt.show()

    is_proccesed_image = False
    bin_pred_converted = False
    multi_pred = False
    found_nose = ''
    img_text_lst = []
    processed_image = ''
    processed_image_lst_converted = ''
    
    if selected_task == "classImageUpload":
        if image_process == 'binaryClass':
            bin_pred = hlprs.binary_class_pred(rgb_image_array,bin_model_name)
            bin_pred_converted = float(bin_pred)
        if image_process == 'multiClass':
            multi_pred = hlprs.multiclass_clas(rgb_image_array,multi_model_name)
            multi_pred = [(id, label, float(prob)) for id, label, prob in multi_pred]
        if image_process == 'objectDetection':
            processed_image = hlprs.object_detection(rgb_image_array,detection_model)
            is_proccesed_image = True
    
    if selected_task == 'segImageUpload':
        
        processed_image, found_nose = hlprs.custom_seg_model(rgb_image_array)
        is_proccesed_image = True

    if selected_task == 'ocrImageUpload':
        processed_image_lst, img_text_lst  = hlprs.img_to_text(rgb_image_array,'image/jpeg')
        processed_image_lst_converted = []
        _, buffer = cv2.imencode('.png', processed_image_lst[0])
        processed_image = base64.b64encode(buffer).decode('utf-8')
        processed_image_lst_converted.append(processed_image)
        print('PRCOESSED THE IMAGE')
        
        is_proccesed_image = False

    # Convert the NumPy array to a base64-encoded string
    if is_proccesed_image: 
        _, buffer = cv2.imencode('.png', processed_image)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        print('PRCOESSED THE IMAGE')
        
    else:
        _, buffer = cv2.imencode('.png', rgb_image_array)
        processed_image = base64.b64encode(buffer).decode('utf-8')

    
    pickle_file_path = 'image_data_after.pickle'
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(processed_image, file)


    
    # Dummy response for demonstration purposes
    response = {'status': 'success','img':processed_image, 'binPred':bin_pred_converted,'multiPred':multi_pred, "foundNose":found_nose,'processed_image_lst':processed_image_lst_converted,'imgTextOCR':img_text_lst}
    print('multi_pred',multi_pred)
    print('bin_pred_converted',bin_pred_converted)
    return jsonify(response)


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

    if image_process == 'identityKernel':
        return                 


    # print(image_data)
    pixel_data = image_data
    pixel_data = np.array(list(pixel_data.values()))
    
    red_array = pixel_data[2::4]
    green_array = pixel_data[1::4]
    blue_array = pixel_data[0::4]

    # Combine the R, G, and B arrays into a 3-channel 2D array
    rgb_image_array = np.stack((red_array, green_array, blue_array), axis=-1).reshape(image_height, image_width, 3).astype(np.uint8)
    
    # Save the image data as a pickle file
    pickle_file_path = 'image_data_before.pickle'   
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(rgb_image_array , file)
    
    histr = ''
    amplitude_threshold = ''
    semantic_img = False
    
    # Process the image data in your Python script
    print(image_process)
    if image_process == 'resize':
        image_data_array_edited = hlprs.resize_image(rgb_image_array,image_width_selected,image_height_selected)
        print('imageRESIZED')
    if image_process == 'translate':
        print('TRANSLATE')
        image_data_array_edited = hlprs.translate_image(rgb_image_array,translate_distances)
    if image_process == 'affine':
        image_data_array_edited = hlprs.affine_transformation(rgb_image_array, image_affine_transform)
    if image_process == 'swapColour':
        image_data_array_edited = hlprs.reconstruct_image(image_data, image_width, image_height, image_current_colour_scheme, image_desired_color_choice)
        # image_data_array_edited = hlprs.swap_colour(pixel_data,image_desired_color_choice,image_current_colour_scheme)
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
    if image_process == 'sharpeningKernel':
        image_data_array_edited = hlprs.sharp_kernel(rgb_image_array,image_selected_kernel)
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
    if image_process == 'identifyShapes':
        image_data_array_edited = hlprs.identify_shapes(rgb_image_array)        
    if image_process == 'FftSpectrum':
        image_data_array_edited = hlprs.fourier_spectrum_20(rgb_image_array)
    if image_process == 'FftFilter':
        image_data_array_edited, amplitude_threshold = hlprs.fourier_threshold_inverse(rgb_image_array, image_fft_Filter_Selection,int(image_slider_output)) #fft_threshold
        # Convert the amplitude_threshold to a base64-encoded string
        _, buffer = cv2.imencode('.png', amplitude_threshold)
        amplitude_threshold = base64.b64encode(buffer).decode('utf-8')
    if image_process == 'edgeDetection':
        image_data_array_edited = hlprs.edge_detection(rgb_image_array,image_selected_edge_detection)
    if image_process == 'threshSeg':
        image_data_array_edited = hlprs.thresh_clust(rgb_image_array,int(image_slider_output))
    if image_process == 'clusterSeg':
        image_data_array_edited = hlprs.img_cluster_segmentation(rgb_image_array,image_cluster_seg,int(image_slider_output))
    if image_process == 'watershed':
        image_data_array_edited = hlprs.watershed_segmentation(rgb_image_array)
    if image_process == 'semantic':
        global df_img_seg, seg_model_results
        semantic_img = True
        image_data_array_edited,df_img_seg, seg_model_results  = hlprs.img_segmentation(rgb_image_array)
        image_data_array_edited = image_data_array_edited[..., ::-1]


    # image_data_array_edited[..., [0, 2]] = image_data_array_edited[..., [2, 0]]


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
    response = {'status': 'success', 'img': image_data, 'desiredColourScheme':image_desired_color_choice,
                 'histogramVals': histr, "fftThresh": amplitude_threshold, 'semanticBool':semantic_img}
    return jsonify(response)


if __name__ == '__main__':  
    
    app.run()


#http://127.0.0.1:5000
    
