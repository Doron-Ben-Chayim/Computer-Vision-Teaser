# COMPUTER VISION TEASER
## By Doron Ben Chayim

Hi,
Welcome to the Computer Vision Teaser a Website that is Designed to help people learn and interact with some computer vision topics. Computer Vision (CV) is a vast an excitng field, and can be somewhat overwelming, so this website is designed to slowly guide the user through various topics, starting with simpler techniques and eventually arriving at more complicated Data Scince Algorithms.

You wont need to scoure the internet looking for CV concepts, and then spend the time coding to implement them, only to find that if you wanted to combine other ideas, it would require more coding and then its a mess, since I have done that for you and even removed the mess. So hopefully through using this website in conjunction with this ReadMe, you will be able to quickly learn and experience useful and interesting CV ideas. 

For a quick video tutorial on how to use this website click [here](https://youtu.be/Jx74ANQuN_E)

### How to Use this ReadMe
This ReadMe provides details on each Main Topic and Subtopic within the website. Below you will be able to find a simple outline of the concepts, a more technical outline of what the algorithm is doing and advice on how to best experience these concepts through suggesting which pictures to use. The pictures can be found [here](https://github.com/Doron-Ben-Chayim/Computer-Vision-Teaser/tree/main/static/user_images).

## Table of Contents

- [How To Use Website](#how-to-use)
- [Quick Outline of Images in Computers](#how-to-use)
- [Basic Operations](#basic-operations)
  - [Resizing](#resizing)
  - [Cropping](#cropping)
  - [Translating](#translating)
  - [Swapping Colour Scheme](#swapping-colour-scheme)
  - [Grayscale Conversion](#grayscale-conversion)
  - [Image Rotation](#image-rotation)
  - [Affine Transformation](#affine-transformation)
- [Enhancement and Preprocessing](#enhancement-and-preprocessing)
  - [Simple Thresholding](#simple-thresholding)
  - [Adaptive Thresholding](#adaptive-thresholding)
  - [Otsu Thresholding](#otsu-thresholding)
  - [Image Histogram](#image-histogram)
  - [Histogram Equalization](#histogram-equalization)
- [Contours](#contours)
  - [Draw Contours](#draw-contours)
  - [Contour Features](#contour-features)
  - [Bounding Boxes](#bounding-boxes)
  - [Identify Shapes](#identify-shapes)
- [Kernel Selection](#kernel-selection)
  - [Identity Kernel](#identity-kernel)
  - [Smoothing/Blurring Kernels](#smoothing-blurring-kernels)
  - [Sharpening Kernels](#sharpening-kernels)
  - [Edge Detection Kernels](#edge-detection-kernels)
  - [Morphological Kernels](#morphological-kernels)
  - [Custom Kernel Tool](#custom-kernel-tool)
- [Fourier Transform](#fourier-transform)
  - [Show Spectrum](#show-spectrum)
  - [Filter](#filter)
- [Edge Detection](#edge-detection)
  - [Sobel](#sobel)
  - [Canny](#canny)
  - [Prewitt](#prewitt)
  - [Robert Cross](#robert-cross)
  - [Laplacian](#laplacian)
  - [Scharr](#scharr)
- [Image Classification](#image-classification)
  - [Binary](#binary)
  - [MultiClass](#multiclass)
- [Object Detection](#object-detection)
  - [Faster R-CNN](#faster-r-cnn)
  - [YOLO](#yolo)
- [Image Segmentation](#image-segmentation)
  - [Threshold](#threshold)
  - [Clustering](#clustering)
  - [Watershed](#watershed)
  - [Semantic/Instance](#semantic-instance)
  - [Custom Instance](#custom-instance)
- [OCR + Analysis](#ocr-analysis)
  - [Get Text From Image/PDF](#get-text-from-image-pdf)

## How To Use Website
This website allows the user to select a Main Process and Subesequent Subprocess and then when then click/upload/Select Area
the said process will commence and the results will appear shortly. The easiest way to understand how to use this website is by lookig at the following example screenshot that outlines the most common elements you migth see during use (There are more that are not in the image, but they are easy to understand). 
![Diagram](https://github.com/Doron-Ben-Chayim/Computer-Vision-Teaser/raw/main/static/website_images/readMeDiagram.jpg)

1) Read Me Button: If you got this far, then this doesnt need explaining, but if you want to open the readme, click this button. 
2) This is an indication if you are ready to click the image and begin the chosen process. If the light is green you are good to go, click away and wait for the results to appear, if they are red, you need to find what needs to be ammended, which will be highlighted in red aswell elsewere on the website.
3) This is the First Dropdwon list, click it to open the selection of Main Proceses. Click the desired process to continue.
4) This is the Sub Process list, this will open up a list of Sub-Processes which once clicked will show more options to continue the process.
5) This is an example Sub-Process dropdown list, click the desired option to continue. 
6) These are the Selected Sub Process Options, each sub process has different options that will appear and each one has a different functionality. Insert the desired values to see the desired effects, however to continue the values must be valid inputs and the outlines must be green. If not you will be alerted and asked to input the correct values. (Item 12 will be explained shortly) 
7) Click this button to Swap out the Main Image (Item 10). It can be very interesting to see how different images respond to the same algorithms and vise versa. Play around have fun, and dont forget to check out the recomended images per subprocess (elaborated in the respective subprocess) to get the best experiences.
8) This will Reset the main image to the Image that appeared before any changes took place. Once an Image is Uploaded, reseting will revert to the most recent uploaded image. 
9) Download the main image as it currently appaers. Here you can save your interesting creations. 
10) This is the main image. You will be using it to apply all the transformations ane experiments. 
11) This is a Selection Box. There are many ways to choose what part of the image the technique will be applied to, one of which is to select the area by moving a square of desired size of the image and clicking. Whatever is in the purple square will have the technique applied to it. This will be elaborated on further in point 12.
12) This specific box is quite common and allows the user to select a method for selecting what area of the main image will have the technique applied to it. There are 3 option:
    a) The Entire Image: This will be selected if the top button is cliked to be on the right. If it is selected, the entire main image will be proccesssed. If it is the the left, then only a partial amount of the image will be selected. 
    b) Partial: There are two options to select a partial area of the main image:
     - Drag: You can click and drag, the first click will place the firs anchor of the square, allowing you to now drag to another spot in the main image, highlighting the selected area. A second click will snip the desired area and send it off for processing. 
     - Stamp: If you want a specific size and dont want to deal with finicky drag boxes, you can select a "stamp" box, which can have its dimensions selected in one of the options above. Its a simple as selecting the dimensions and choosing what to include buy hovering the square over the desired area and clicking to send whatever is in the square to be proccesed. 
 13) This will show the user what is currently being selected by the purple "Stamp"
 14) After a selection has been made by either draggind or stamping, the processed image will appear below (see 16), to reset the image and allow for move images to appear on the screen, you can click the reset Capture X button. (PLEASE NOTE YOU ARE LIMITED TO THREE CAPTURES)
 15) If you want to download the selceted capture, click this button. 
 16) The selected proccesed area will appear on the right side of the screen. There are 3 rows, with the top row, being capture 1, middle being capture 2 and the last being capture 3. 
## Quick Outline
What is an image in the computer world? Well to put it simply, a coloured image is the combination of atleast 3 layers (AKA Channels) of grids (AKA Matrixes), where each grid has a certain number of rows and columns of squares (AKA Pixels) filled with certain numerical values. The 3 channels generally each focus on a certain colour, Red, Green and Blue respectively and just like mixing differnt coloured paints produces different colours, mixing different amounts of Red, Green and Blue, in the corresponding pixel value can produce different colours. A simple image might be what is known as 8-bit which allows for 256 (derived from 2^8 possible combinations of bits, starting at 0, so effectively in the range 0-255) possible values. If the pixel value is zero that represents the lowest intensity and if it is 255 that is the highest. So for example, you might have an image that has 1920 rows and 1080 colums (1920x1080 resolution), which means there are 2073600 possible pixels per channel. If we combined all the pixels in each channel that are located in the same spot per channel, lets say the one in the top left corner which in matrix notation would be row 0, column 0 (or [0,0],  [row_num, colum_num), there could be a combination of the three values, and that gives us a colour, for example if the Channels are organized in Red, Green, Blue (RGB), the value could be [0,0,0], which would be black as all channels have zero intensity, but if it was [255,255,255], then it would be white as it is the maximum intensity, and lastly if it was [255,255,0] it would be yellow as red and green are at full intensity and blue is at the lowest intensity. This model is based on the additive model of colours, so dont get confused if this contradicts what you learnt in primary school which was probably based on the subtractive model. So know we know that an image is a 3 layered grid, combining different intensities of colours, we can understand that if we change these values or find patterns in them we can effectively transform them to suit our needs. 

## Basic Operations
Recommended Image: Any, but the starting Elephant is great. 
### Resizing
What is resizing?
Resizing is effectivly changing the resolution of the image, either increasing or decreasing the number of rows and columns. How is this Done?
It tries to transfer the certain values that currently exist in certain locations into a future either larger or smaller area that it intends to create, it does this through interpolation/extrapolation through looking at the surrounding pixels and averaging them or filling extra space that didnt exist with the same value around the pixel in consideration. There are many different algorithms (e.g Nearest Neighbor Interpolation, Bilinear Interpolation, Lanczos Resampling ) that can be implemented to do this seemingly simple task and each one has its own strengths and weaknesses. A good example could be making a quilt, where if you want to increase its size you try and match the colours to the ones near to the new squares, and if you want to remove some squares you may have to change the colours to an average of the ones that remain so that it is more cohesive.   

### Cropping
What is Cropping?
Cropping is selecting a certain area within the image.
How is Cropping Done?
Well now that we know that an image has a certain number of rows and columns, cropping involves selection the pixels found in the desired rows and columns. 

### Translating
What is Translating?
Sometimes an image might be in another language . . . im kidding, translating an image is sliding the image over by a chosen amount, meaning that each pixel is moved over a certain number of pixels. All the values stay the same, but they are now in a different location. The image can move in the x (horizontal), y (vertical) or both direction. But you are probably wondering, if everything is moved over, then nothing changes, like if I move a chair from the left side of the living room to the right? Well two things can happen to change the image, 1) Pixels that are pushed out of the origional image dimensions are removed, for example if a pixel was located at [90,90] and we wanted to move everything 20 over to the right, then that pixel would be cut off and removed from the image as it doesnt have anywhere to go. 2) If we move everything over a certain direction then isnt there now empty pixels, where the pixels used to be? The answer is yes, those can be filled with any desired values, but generally filled with black. So continuing with the chair example its as if you effectivly slide the chair to the right past 2 barriers, one will perfectly destroy any part of the chair that passes the barrier, the other one will fill the chair with a black void of nothingness on the other end to ensure that the chair is always the same size.   

### Swapping Colour Scheme
Recommended Image: Pepper
What is Swapping Colour Schemes (or color for our American friends)
As previously explained about images are created through combining the corresponding pixels value in different layers, and there are many different layer compositions and each one also has corresponding values to ensure that the images appear as they should, for example if we needed to buy 1 kg of gold with different currencies, we would need different amounts of the respective currency to account for their intrinsict differnces in value. Here are some common colour Schemes:
1) Red Green Blue (RGB)
2) Blue Green Red (BGR) - Same as RGB but the B and R channels have swapped order. What would be the difference in the two, here is a good example:
RGB: (255, 0, 0) – This means full intensity of red, and no green or blue, resulting in a red color.
BGR: (255, 0, 0) – This means full intensity of blue, and no green or red, resulting in a blue color.
The same pixel values (255, 0, 0) represent different colors depending on whether the format is RGB or BGR.
3) HSV (Hue Saturation Value) :
    - Hue: This represnets the colour and is similiar to an angle that is is betweeon 0 and 360 with 0 being red and 300 being magenta, like a colour spectrum you might see from a prism. 
    - Saturation: This is the vibrancy which is the degree it differs from a natural gray, in percentages, so 0% would be gray aka desaturated and 100% would be Full colour aka fully saturated
    - Value: Represents the brightness/luminance, so how light or dark the colour is, also a percentage 0% is black aka no brightness and 100% is full brightness
    
Converting between HSV and RGB has some complicated formulas and rules, but it can be done. 
You also might be wondering why the image appears strange if it is converted then shouldnt it look the same?

When you convert an RGB image to HSV, each pixel's color information is split into these three components. If you directly visualize the HSV image as if it were an RGB image, you are not seeing the intended colors but rather a misinterpretation of the HSV values. HSV is a format that saves the values like a path you can retrace, but the image that is shown is only half the journey, not a the full trup back to rgb, thats why the image does not look the same. 

Similariily when you swap from RGB and BGR, you are essentially swapping the red and blue channels. This means that what was originally interpreted as red in the RGB image will now be interpreted as blue, and vice versa. This can significantly alter the appearance of the image because the color interpretation has changed.

Example of Channel Swapping
Original RGB Pixel: (255, 0, 0) — This represents a fully red pixel.
Converted BGR Pixel: (0, 0, 255) — This represents a fully blue pixel.
By swapping the channels, you change the color that each pixel represents.

### Grayscale Conversion
Recommended Image: Mandrill
What is Grayscale Conversion?
Grayscale Conversion, similiar to colour scheme conversion, (in the fact that operations are applied to the different channels, changing the final image output), is applying a process to an image to ensure that the final image consists of only different shades of gray. The end result is a one channel image, that is the averaged values of the RGB channels. So for example if you had a pixel with RGB value [66, 135, 245] a nice shade of baby blue, and convert it to grayscale you will get a value of (66+135+245)/3 which is 149, a nice shade of gray. The shade of gray is derived from the new color scheme where 0 is black and 255 is white and all the values in between represnet a different combination of those two colours.  

### Image Rotation
What is Image Rotation?
Image rotation is a geometric transformation that involves turning an image around a central point, usually the image’s center, by a specified angle.The pixel values are the same, however they are all moved to a different location, and just like image transaltion can result in cut off and filled in pixels. This is done ususally by multipplying the image by a "Rotation Matrix"

Here is a good article to explain what is happening:
https://medium.com/street-science/the-math-behind-image-rotation-5e107e5881da

### Affine Transformation
What is Affine Transformation

Affine transformations are ways to change an image that keep straight lines straight and keep distances between points proportional. These changes can include moving the image, resizing it, rotating it, and slanting it. In image processing, affine transformations change where the pixels are but keep the overall structure of the image. This is more flexible than just rotating an image, as it allows for a wider range of changes. In this website you can have the option to change the scale of the image as well as specifying what angle you would like to rotate it by. 

## Enhancement and Preprocessing

### Simple Thresholding
Recommended Image: Thresholding_Gradient
What is thresholding? 
Thresholding is a technique used to change the value of a pixel based on its current value, a predetermined threshold and a set of rules to change the value. For example, in "Binary Thresholding" if you set a threshold value of 127, you can change the pixel's value according to a specific rule: if the pixel's current value is above the threshold, it might be set to 255 (white); if it is below the threshold, it might be set to 0 (black). Depending on whether you want a standard threshold or an inverse threshold, the rule can be adjusted accordingly. In an inverse threshold, the values are swapped: pixels above the threshold are set to 0, and those below it are set to 255.

Another method is known as "To-Zero" Thresholding, where there is also a specific threshold, but instead of setting the pixel to the maximum or minimum value, only pixels that are below the threshold are set to zero. Pixels above the threshold retain their original value or vice versa. This method is useful when you want to preserve the intensity values of pixels above the threshold.

This method is commonly used in image processing to create binary images, making it easier to analyze and process further. This is a step that can be implemented to allow for easier analysis later on, it usually can clear up noise, help isoalte features, and can generally improve the performance of complicated algorithms. 

### Adaptive Thresholding
What is Adaptive Thresholding?
Sometimes the global threshold set in simple thresholding is not sufficient pre-processing and a more complex approach is required to reach a more refined output. Adaptive thresholding finds many different thresholds using by comparing each pixel to its surrounding pixels, using the neighbours to determine many "local" thresholds. In the website you can experiment with two different methods:
    1) Mean: The threshold value is the mean value of the surrounding pixel densities (Minus a Constant)
    2) Gaussian: The threshold value is a weighted sum of the pixel values, where the weights follow a gaussian distribution, meaning pixels closer to the pixle being investigated get more importance. 

In the website besides for the method of adaptive thresholding you will also be able to choose:
    - Max Pixel Value: This value determines what the pixel will be set to if it meets the threshold condition.
    - Threshold Type: Explain in the above paragraph, but you can chooses between "binary" and "inverse Binary"
    - Block size: The block size defines the width and height of the square neighborhood around each pixel. For instance, a block size of 11 means an 11x11 neighborhood, with the pixel in question being in the centre.
    - Constant: The constant C is a value that is subtracted from the computed mean or weighted sum of the local neighborhood of pixels. This constant is used to fine-tune the thresholding process, allowing for better control over which pixels are classified as foreground or background.
Details about adaptive thresholding.

### Otsu Thresholding
What is Otsu Thresholding?
Otsu Thresholding is a technique that tries out different threshold values to divide an image into two groups. At each value, it analyzes the variance level within the two groups and then decides on a threshold that results in the lowest variance within each group. The two groups are known as "foreground" and "background" (this has nothing to do with physical depth; it is based on pixel intensity values), with the foreground generally being the more important aspect of the image. Otsu's Thresholding is named after Nobuyuki Otsu, a Japanese researcher who developed this method. 

### Image Histogram
What is an Image Histogram?
A histogram is a graphical representation of the distribution of numerical data. It displays the frequency of data points within specified ranges (bins) and is used to provide a visual interpretation of the data's distribution. So when we produce an image histogram we are looking at the frequency distribution of the values per channel. So we would get three lines, one for each colour channel. The patterns that appear in the histogram can be useful to understand the characteristics of the image such as:
    - Contrast: A wide spread could mean a high contrast.
    - Brightness: A left or right skew could show a bright or dark image.
    - Color Distribtuin: It can show what are the predominant colours, which can help with other proceses like Content-Based Image Retrieval. 

### Histogram Equalization
Histogram equalization is a technique used in image processing to improve the contrast of an image. It can helo with feature detection and general visual quality. This method enhances the intensity contrast by spreading out the most frequent intensity values, making it easier to distinguish different features in the image. In short the histograms Cumulative Distribution Function (CDF) is normalized and the origional values are then mapped to the new values. 

## Contours
Recommended Image: The image titled "Contour_Shapes" or "Coins"

What are contours? Contours are curves that connect the boundaries of areas that have similiar values. For example a contour might be a line around a square within a shape, or it could be a more complex outline depending on the defenitions required. 
### Draw Contours
This will draw the contours (outlines) of any shapes that it detects. While this appears to be a simple process, there are quite a few different complex algorithms involved but one of the main ones is the Suzuki-Abe algorithm. In short this algorithm, binarizes the image, then begins iterating throguh the pixels, marking them as part of the contour or not depending on the value present in the pixel. 

### Contour Features
Once a contour has been found there are some characterstics each one has:
    - Perimeter: This is not just the number of pixels in the perimeter, but it is the sum of the euclidean distance between succesve contour points.
    - Area: This is the number of pixels contained within the contour.
    - Centre: This is the the centre of the contours, calculated by looking at the different moments of the contour. some moments include the Zeroth Moment (total number of pixels). First Moment (centre of mass), Second Moments (Describes the orientation and spread.)

### Bounding Boxes
A bounding box is the smallest possible shape, such as a rectangle, circle, or ellipse, that fully encompasses a contour.In this website you can play around with 4 shapes, a rectangle (aligned with the images axis), an alligned rectangle (more aligned with the contours direction), a circle and an ellips.

### Identify Shapes
This section allows you to label the contours with their respective shapes, a practical outcome of finding the contours. How does the computer infer shapes from the contours? Well just like you and I would, it tries to find the number of corners, the ratio of the lengths of the sides and any other common properties you might expect to find in a shape. There are many different approaches.  

## Kernel Selection
Recommended Image: "Woman_in_Hat"
A kernel in the context of computer vision is a square matrix of odd dimensions that effectively slides (also known as convolves) over the image. At each step, it multiplies itself (performing a dot product) with the pixel values beneath it, resulting in a different image depending on the kernel's values. A kernel is also known as a filter, mask, or convolution matrix. It should not be confused with a colonel in the army, a corn kernel stuck in your teeth, or the kernel of a computer's operating system.

There are many different options when it comes to convoling a kernel over an image (AKA Convolution), which can drastically change the outcome of the final image, making it sharper, blurrier, thinner, highlighting edges ect. Some of the choices include:
    - Size of the Kernel: 3x3, 5x5 ect 
    - Contents of the Kernel: Will the number be small, big, will there be a certain pattern for example top left triange filled with certain values, or will it be more of a cross shape, the possibilites are endless.
    - Stride: How many pixels does the kernel move at each step.
    - Padding: Sometimes the image has extra layer(s) added, with different values to account for the unique examples that arrise with certain shaped images combines with their kernel size and stride size. 

I wont be showing each kernel here, just a bried outline of the basic characterstics, for a visual reprentation of the different types of kernels please choose the "Custom Kernels" Sub-option and open the Custom Kernel Building Window by clicking the button.   

### Identity Kernel
The identity kernel is a special type of kernel used in image processing that, when convolved with an image, leaves the image unchanged. It effectively acts as a "do nothing" filter. This kernel is particularly useful for testing and understanding the effects of other kernels because it provides a baseline output.

### Smoothing/Blurring Kernels
Smoothing or blurring kernels are used in image processing to reduce noise and details, creating a smoother or blurred effect in an image. These kernels work by averaging the pixel values within a neighborhood, which softens edges and reduces intensity variations. Commone kernels include:
    - Box Blur: This kernel replaces each pixel with the average of its neighboring pixel values.
    - Gaussian Blur: This kernel uses a Gaussian function to calculate the weights, giving more importance to the central pixels.
    - Median Blur: This is not a kernel in the traditional sense but a nonlinear filter that replaces each pixel with the median value of its neighborhood. Thats why it cant be made in the kernel tool, as it requires knowledge of the image. 
    - Bialteral Filter: 

### Sharpening Kernels
Sharpening kernels are used in image processing to enhance the edges and fine details of an image, making it appear more defined and crisp. These kernels work by emphasizing the differences between adjacent pixel values, which helps to highlight edges.

- Basic Sharpening: 
Usually has negative edges and positive centre,which subtracts the surrounding pixel values from the central pixel value and then adds the result back to the central pixel value.
- Laplacian Kernel: Similiar to basic sharpening, however the central value is negative with the surrounding being positive, this highlights regions of rapid intensity changes and is good at finding edges. 
- Unsharp Masking: Involves subtracting a blurred version of the image. 

### Edge Detection Kernels
Edge detection kernels are used in image processing to identify and highlight regions in an image where there are significant intensity changes, typically corresponding to object boundaries. These kernels work by detecting gradients or changes in intensity in different directions.Common edge detection kernels include:

- Sobel: The Sobel operator detects edges by calculating the gradient of the image intensity in both the x and y directions. Sometimes the output of the x and y sobel can be combined to allow for edge detecion in both axis. 
- Prewitt: Similar to the Sobel operator but with different weighting factors.
- Laplacian: Described above. 
- Scharr: Some may argue that this is an improvement over the Sobel operator. It provides a more accurate approximation of the derivative by giving more weight to the central differences. This results in better edge detection, especially in situations where precision is critical. The main difference is that this has higher values on the edges to help emphasize the differences more. 

### Morphological Kernels
Recommended Image: "J", 'Closing', "Opening"
Details about morphological kernels.

### Custom Kernel Tool
This tab allows you to visualize common kernels, and then apply them to the image. Select the kernel you want, and then press the "Click Me". On the main page you will recieve a visual clue that the kernel has been updated to what you selected in the window. Click the image to apply the convolution and see the end result. In addition to the preloaded kernels, which can be changed, there is also an option to create your own kernel from scratch to play around with and see what will happen. Have fun. 

## Fourier Transform
Recommended Image: "CameraMan"
The Fourier Transform is an absolutely amazing algorithm that is quite complicated—so much so that I don't think anyone actually understands it. If someone says they do, they're probably lying.

What the Fourier Transform does is swap a signal from the time domain to the frequency domain. This is extremely useful in many fields, as it represents data in a way that can reveal valuable information, like the underlying frequencies in a signal. The real beauty, however, lies in the fact that there's also the inverse Fourier Transform, which can transform the signal back from the frequency domain to the time domain. Ok but what is the time domain and what is the frequency domain?

- Time Domain: In a traditional signal like a sound recording, this shows how a signal changes over time. For images, it means how pixel values vary across the picture, like a 2D grid of brightness levels. This is the normal way we see the original image. 
- Frequency Domain: Represents a signal by decomposing it into its constituent frequencies. It shows how much of the signal lies within each given frequency band over a range of frequencies. In the frequency domain, signals are typically visualized as spectra, showing the amplitude (or power) of each frequency component.
- Low frequencies correspond to slow changes or smooth variations in pixel intensity
- High frequencies correspond to rapid changes in pixel intensity values

In image processing, this is done through implementing the Discrete Fourier Transform (DFT), a variation of the formula outlined above, which can be efficiently implemented on computers using the Fast Fourier Transform (FFT) algorithm. This algorithm drastically reduces the computational runtime, making the process feasible even for large datasets.

### Show Spectrum
What is this gray image that is shown after applying the Fourrier Transform? To put it simply try imagine that the pixles are not being placed in the same corresponding spot they are found in the origional image, but rather the new image is a representation of the frequency information about each pixel organised in a way, that higher frequencies can be found towards the centre of the image and lower frequences can be found around the edges (some people swap that order). The brightness seen, represents the magnitude (quantity) of each frequency, with brighter colours represnting a higher mangitude and a darker colour represnetring a smaller magnitude. A good analogy is imagine someone was stacking up piles of money, the higher denominations like the 100 dollar bill need to be placed in the centre and the smaller denominations needs to be placed on the edges, and the height of each pile (the brighntess in our example) shows how much you have of each denomination. The axis represente these frequency values, represented in cycles per minute. This can show us:

1. Central Bright Spot 
Pattern: A bright spot at the center of the spectrum.

Indicates: Strong high-frequency components.
Interpretation: The image has fine details, edges, or rapid changes in intensity.
Example: An image with a lot of sharp edges or intricate patterns.
2. Horizontal and Vertical Lines 
Pattern: Bright horizontal and/or vertical lines extending from the center.

Indicates: Strong low-frequency components along one direction.
Interpretation: The image contains broad, smooth variations or repetitive patterns aligned along the horizontal or vertical axis.
Example: A gradient or slowly varying texture in the image.
3. Diagonal Lines 
Pattern: Bright diagonal lines.

Indicates: Diagonal low-frequency components.
Interpretation: The image has smooth variations or patterns that repeat diagonally.
Example: A gradual diagonal gradient.
4. Isolated Bright Spots
Pattern: Bright spots near the center.

Indicates: Specific high-frequency components.
Interpretation: The image contains fine details or edges at particular orientations.
Example: Detailed textures or fine patterns, such as leaves on a tree or fabric weave.
5. Ring Patterns
Pattern: Concentric rings around the center.

Indicates: Radial symmetry or circular patterns.
Interpretation: The image has features that repeat radially.
Example: Ripples on water or radial gradients.
6. Complex Patterns
Pattern: Intricate, non-uniform patterns.

Indicates: Complex textures and details.
Interpretation: The image contains a mix of different textures, edges, and details.
Example: Natural scenes with varied textures like foliage, sand, or fabric.


### Filter
The beauty of the frequency domain is that it allows for powerful manipulations of an image's frequency components. By selectively removing or enhancing specific frequencies, we can achieve various effects such as noise removal, edge smoothing, and edge sharpening. This is possible through the use of the inverse Fourier Transform, which converts the manipulated frequency domain data back into the spatial (time) domain.In this website you have the option to apply a high pass and low pass filter as well as the gaussian equivalent. 

- Low Pass Filter: A low-pass filter allows low-frequency components to pass through while reducing high-frequency components. It is used for smoothing or blurring an image by removing sharp edges and noise.
- High Pass Filter: A Hig Pass filter llows high-frequency components to pass through while attenuating low-frequency components. It is used for sharpening an image by enhancing edges and fine details.
- Gaussian Filter: A Gaussian filters are a more sophisticated version of low-pass and high-pass filters. They use a Gaussian function to weight the frequencies, providing a smooth transition rather than a hard cutoff.

## Edge Detection

### Sobel
Please see Sobel kernel Above

### Canny
This is a multistage Algorithm that does the following steps:
    -1) Gaussian Filter: Apply a Gaussian blur to smooth the image and reduce noise
    -2) Gradient Calculation: Either useing Sobel, Prewitt, or Roberts operators to find the gradient magnitude and direction.
    -3) Non-Maximum Suppression: Thin Edges: Suppress pixels that are not at the local maximum, leaving only the pixels at the peak of the gradient, creating thin edges.
        This entails roughly:
         - Gradient Calculation
         - Local Maximum Check: For each pixel, we check if it is a local maximum in the direction of the gradient 
         - Suppress Non-Maximum Pixels: If a pixel is not the highest value in its gradient direction, it is suppressed (set to zero). Only the highest pixels (the local maxima) are kept, making the edges thin and sharp.
    -4) Apply Two Thresholds: Use a high and a low threshold to identify strong, weak, and non-relevant pixels. Strong edges are sure edges, while weak edges are potential edges.
    5) Final Edge Determination: Strong edges are included in the final output. Weak edges are included if they are connected to strong edges.

### Prewitt
Please see Prewitt kernel Above

### Robert Cross

Details about the Robert Cross method.

### Laplacian
Please see Laplacian above. 

### Scharr
Please See Scharr Above

## Image Classification
Image Classification is an exremely fascinating and practical field of Data Science. This allows us to begin categorizing what is in an image, from the more general approach to catergorizing an image as a whole into one class, to then categorizing many potential classes within an image, to isolating objects within the image, and even providing a class to every pixel in the image, each process is exciting and can be done in multiple different ways. However this website looks at a few genereal topics:
    - Object Binary Classification
    - Object Multi-Class Classification
    - Object Detection
    - Image Segmentation
    - Optical Charatacter Recognition (OCR)

Here is a image that should be able to help visualize these different process:
![Image Classification Outline](https://github.com/Doron-Ben-Chayim/Computer-Vision-Teaser/raw/main/static/website_images/image_classifcaiton_diagram.jpg)

Behind the scenes, how does a computer know how to classify an image it has never seen? The answer is that it may not have seen this exact image, but if you show it enough similar ones, it can learn patterns. When it identifies those patterns again, it can say that this image is most likely "X" since "X" has the same patterns that it has seen before. This is what is known as machine learning: the computer is learning to find patterns in the data and then match those patterns elsewhere.

In the case of computer vision, there is a system in place where the computer can learn different kernels. Yes, the kernels we have been dealing with so far can be learned. To put it simply, there is an architecture where an image is provided as input. Then, hundreds of these kernels convolve over it. With a few more processes, such as non-linear activations and pooling, we arrive at a final output. We compare the computer's output with what it should be. If there are errors, it goes back (through a process called backpropagation) and changes the kernel values according to how wrong it was (using an optimization algorithm like gradient descent). Eventually, after going backwards and forwards (forward pass) many times with many different images, it refines the kernel values so that the processing that happens to the image will most likely produce the correct classification.

In summary, the computer learns to identify patterns by processing many images and adjusting its parameters through backpropagation and gradient descent. Over time, this enables the computer to accurately classify images it has never seen before based on the patterns it has learned. These architectures are a type of Neural Networks (NN) called Convolutional Neural Networks (Cnns).

Some common databases that people train images on:
    - ImageNet: Has 14 million images in 20,000 categories. 
    - CIFAR-10 and CIFAR-100:  CIFAR-10 contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. CIFAR-100 is similar but has 100 classes with 600 images each.
    - MNIST/Fashion MNIST: A database of 70,000 handwritten digits (0-9), where each image is 28x28 pixels and 70000 grayscale images of 10 fashion categories. 
    - COCO (Common Objects in Context) : Contains over 330,000 images, with more than 200,000 labeled for object detection, segmentation, and captioning.

### Binary
Recommended Image: 'Cat' 'Daisy'
Binary Classifcation, as the name implies is classifying the entire image into one of two classes. In this case I have gone with the classic "Dog vs Cat" challange, where the model is trained to predict wheather or not the new image is of a dog or cat, not bear in mind, it can only say cat or dog, so if you put in a picture, no matter what it is (ferrari, table ect) it will say that it is either a cat or a dog. But what are the three options available?
    - Custom Model: This is a model that was built and trained by myself, where it was shown only the kaggle "Cat Dog" Dataset. 
    - Vgg16: VGG16 is a convolutional neural network (CNN) architecture proposed by the Visual Geometry Group (VGG) at the University of Oxford. It achieved state-of-the-art performance on the ImageNet dataset. Using a process knowen as transfer learning, the weights of this model can be "frozen" and the training process can begin again this time looking at just the cat and dogs images with the final layer of the model being trained. This is beneficial in multuple ways as it can save time, money and provide better results. It is the literel realization of the saying "stanind on the shoulders of giants".
    - ResNet: (Residual Network) is a deep convolutional neural network (CNN) architecture introduced by Microsoft Research. 
    Why are there three models? The answer is that each one has a different architecture, such as the number of layers, the size of the kernels, the processes that happen between each one, what dataset they were trained on, efficiencies ect, as such each one can be better suited to some tasks then others. 

### MultiClass
Recommended Image: "Cat" "Daisy"
Multi-class classification is a step up un complexity to binary classifcation as it classifies the entire image inot many potential classes. The classes that it can classify into are limited by the classes it saw during training, therefore to allow for more class options, there needs to be a larger dataset and more time training which takes time and money. There were two models used in this website:
    - Xception (Extreme Inception): Introduced by François Chollet, the creator of Keras, Xception is an extension of the Inception architecture that replaces the standard Inception modules with depthwise separable convolutions.
    - InceptionV3: An improved version of the original Inception architecture (also known as GoogLeNet), introduced by researchers at Google. It includes several enhancements over its predecessors.
    
Other famous and accesible pre-trained multi-class models are EfficientNet, ResNeXt, DenseNet, MobileNetV2

## Object Detection
Recommended Image: "yolo_1" "yolo_2"
Object detection is a more complex task that attempts to detect and outline specfici object within and image. There are many differnet algorithms designed to tackle this problem, two were used in this website. 

### Faster R-CNN
Faster R-CNN stands for Faster Region-based Convolutional Neural Network. It is a model used to detect objects within an image, introduced by researchers Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun in 2015. The model begins by using a network like VGG16 or ResNet to scan the image and create a detailed map of its features, identifying interesting patterns and textures. Then, a Region Proposal Network (RPN) looks over this feature map to find potential regions where objects might be located, using small sliding windows to make initial guesses about the object type and location, drawing rough boxes around them. These proposed regions are then standardized to a fixed size through Region of Interest (RoI) Pooling, ensuring all regions are of the same size for easier analysis. Finally, these standardized regions are fed into another part of the network, which makes more accurate predictions about the objects within those regions, determining what each object is and refining the bounding boxes around them. In summary, Faster R-CNN is a system that identifies areas in an image that might contain objects and then accurately classifies and localizes those objects.

It is called faster r-cnn as it is an improvement on its predecessors, r-cnn and fast-rcnn. 

### YOLO

YOLO, which stands for "You Only Look Once," is a real-time object detection system introduced by Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. Unlike traditional methods that require multiple steps to detect objects, YOLO simplifies the process by looking at the entire image in one go.

Here's how it works in simpler terms:

YOLO divides the input image into a grid. Each grid cell is responsible for detecting objects within that cell. For each cell, YOLO predicts a set of bounding boxes along with their confidence scores, which indicate the likelihood of objects being present and how accurate the bounding boxes are. It also predicts the class probabilities for each detected object.

The unique aspect of YOLO is that it processes the entire image in one pass through the network, making it extremely fast. This allows YOLO to achieve real-time object detection, which is beneficial for applications requiring quick processing, such as video surveillance and autonomous driving.

In summary, YOLO is a system that detects objects by dividing the image into a grid, predicting bounding boxes and class probabilities for each grid cell in a single pass through the network, enabling fast and accurate real-time object detection.

## Image Segmentation
Recommended Image: 'Yolo_1', 'Yolo_2','coins'
Image Segmentation partitioning an image into multiple segments or regions, each representing a different object or part of an object within the image. The 3 most common types of Image Segmentation that use Deep learning are:
    - Semantic Segmentation: Assigns a class label to each pixel in the image, meaning that pixels with the same label belong to the same object or region.
    - Instance Segmentation: Similar to semantic segmentation, but it also differentiates between individual instances of objects. For example, it not only identifies all cars but also distinguishes between each car.
    - Panoptic Segmentation: Combines semantic and instance segmentation, providing both pixel-level classification and instance-level differentiation.
    
This website also looks at simpler methods of segmentation which rely less on machine learning and more on traditional algorithms. 

### Threshold
This method divides the image into segments depending on their intensity value. By selecting the number of thresholds, you are effectivly selecting the number of threshold ranges to divide the image into, whith each pixel being assigned to a certain group.  The thresholding in this website is designed to ensure that each threshold has the same range, meaning if you select 3 groups, then there will be three groups to segment the pixels into Pixels with intensity values from 0 to 84 are assigned to group 1. Pixels with intensity values from 85 to 169 are assigned to group 2. Pixels with intensity values from 170 to 255 are assigned to group 3.

### Clustering
Clustering is a very important and ubiquitous task in data science and can also be applied to images. Clustering tries to group elements based on a defined similarity, in this case, pixel intensity. There are two methods used in this website:

- K-means: This algorithm partitions the image into K clusters. Each pixel in the image is assigned to the nearest cluster center based on the pixel's intensity values. The K-means algorithm iteratively refines the positions of the cluster centers to minimize the variance within each cluster. This results in segments of the image that share similar pixel intensities, which can be useful for tasks such as image segmentation and object detection.

- Mean Shift: This is a non-parametric clustering method that does not require specifying the number of clusters in advance. Instead, Mean Shift clusters the image based on the density of pixels. It works by iteratively shifting each pixel towards the mode of the points' density function. This algorithm is particularly effective for identifying clusters of arbitrary shape and can capture complex structures within the image.

### Watershed

The Watershed method is a popular image segmentation technique that treats the image like a topographic surface where pixel values represent the elevation. It is particularly effective for separating overlapping objects in an image.

Watershed: This algorithm starts by identifying the local minima in the image, which correspond to the lowest points or "catchment basins." The algorithm then simulates the process of water flooding the basins, with barriers being built where waters from different basins meet. These barriers form the watershed lines, effectively segmenting the image into distinct regions. The Watershed method is highly effective for images with varying intensities and complex structures.

### Semantic/Instance

Image segmentation is a crucial task that involves partitioning an image into segments that represent meaningful objects or regions. There are two main types of segmentation: semantic segmentation and instance segmentation.

 - Semantic Segmentation
Semantic segmentation involves classifying each pixel in an image into a predefined category. The goal is to assign a label to every pixel so that pixels with the same label belong to the same object class. However, semantic segmentation does not differentiate between different instances of the same class. For example, in an image containing multiple dogs, semantic segmentation will label all dog pixels as "dog" without distinguishing between individual dogs.

- Instance Segmentation
Instance segmentation goes a step further by not only classifying each pixel into a category but also differentiating between different instances of the same category. This means that each object instance is separately identified and segmented. In the case of an image with multiple dogs, instance segmentation will label each dog individually.

The model used in this website is YoloV8, here you can click the image you want to segment and the image will be processed. Once all the instances have been found, the results will be displayed in an interactive table on the right. Here you will have the option to filter out instances according to confidecne level as well as class. You can also decide on the feature of each instance to show. You can see the outline, the bounding box, the mask and the cutout. Once you have made your selection click "Process Selected" to view the changes. 

### Custom Instance
Recommended Image: "Daisy"
While the previous model is quite powerful, sometimes you need a model that can find specific items in a specific use case and the model may not have been trained to do that. This could be to find insects in produce or tumours in brain scans. These models can be trained and are an extension of the previous model. They use the pretrained weights but they are exposed to more specific use case images in a process known as transfer learning. This website has a custom model that has been trained to find the nose on a dog. This model is for illustrative purposes only and therefore may not be the most amazing model. Generally, to make a model that can be more reliable take hundreds to thousands of photos, where each one is labeleld. This is a passion project, so that was not done, but have fun with the smaller limited version to help learn about the concept :). 

## OCR + Analysis
Recomenended image: "ocr", 'Doron_Ben_Chayim_CV.pdf'
Optical Character Recognition (OCR) is a technology used to convert different types of documents, such as scanned paper documents, PDFs, or images captured by a digital camera, into editable and searchable data. For the OCR tasks in this project, Tesseract was utilized as the primary model. Tesseract is an open-source OCR engine that supports a wide variety of languages and is highly regarded for its accuracy and efficiency. It processes images to extract textual information, converting pixel data into machine-encoded text. By integrating Tesseract, the project benefits from its robust capabilities in handling various text recognition challenges, making it a reliable tool for digitizing and analyzing text-based image data.

In this website you will have the opportunity to scan either images or pdfs. The results will appear on the right hand side of the screen, with two windows appearing, the left one will be the scanned image/page and the right will contain the extracted text. You can click the arrows above the images to swap between pages in the pdf. Underneath the two windows are some checkboxes that can be selected to send the required text to ChatGPT, where you can analyse the text. Please insert the question you want to ask about the text in the designated box, and also add your own ChatGPT API key. Dont worry, nothing will be saved, I just wont be providing my own API key. When you are ready, click "Ask ChatGPT" to get a response to your question. 

