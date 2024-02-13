// Set Variables
let selectedArea = false;
let imageData;
let imageDataHeight;
let imageDataWidth;
let secondDropDownChoice = "";
let savedInitialImage = false;
let initialImage = document.getElementById("mainImage").src;
let selectedAreaType;
let usedAspectRatio = false;
let aspectRatio;
let selectedImageWidth; 
let selectedImageHeight;
let selectedRotateAngle;
let colorChoiceVar;
let currentColourScheme = 'rgbColour'
let tempCurrentColourScheme;
let selectedSimpleThresholdMethod;
let thresholdValue = 127;
let thresholdMax = 255;
let translateDistances = [50,30];
let affineTransformChoices = [45,1.5];
let adaptiveParamaters = []

// Updates the Main Image to return to when Reset Chosen
function setNewInitialImage() {
  // Save the initial image to be used when user clicks Reset Image
  const mainImageElement = document.getElementById("mainImage");  
  initialImage = mainImageElement.src;
  savedInitialImage = true
  currentColourScheme = 'rgbColour'
  }

// Resets the Main Drop Down Menu
function resetMainDrop () {
  // Reset the dropdowns to the initial default value
  const mainDropdownElement = document.getElementById("mainDropdown");
  mainDropdownElement.selectedIndex = 0;
}

// Resets the Secondary Drop Down Menu
function resetAllSecondDrops() {
  const secondDropdowns = document.querySelectorAll(".secondDropSelect");

  secondDropdowns.forEach(dropdown => {
    dropdown.selectedIndex = 0;
  });
}

// Hides the Divs created from the secondary dropdown choices
function removeSecondaryChoiceDivs() {
  const secondChoiceDivsElements = document.querySelectorAll(".subForm");

  secondChoiceDivsElements.forEach(element => {
    element.style.display = 'none';
  });
}

// Hides the Area Selected Div and resets the choices
function removeAreaChoice () {
    //Hide area Elements
    const selectedCategory = document.getElementById("areaSelection"); 
    selectedCategory.style.display = "none";
    selectedArea = false;
    selectedAreaType = "";
}

// Shows the Area Selected Div
function showAreaChoice () {
  //Hide area Elements
  const selectedAreaType = document.getElementById("areaSelection"); 
  selectedAreaType.style.display = "flex";
  selectedAreaType.style.flexDirection = "column";
  selectedAreaType.style.justifyContent = "centre";
  selectedAreaType.style.margin = "5px";
}

// Hides the Hover box size choice div
function showHoverSize () {
  // Hide size of Hover Square
  const hoverSizeSelector = document.querySelector("#hoverSizeSelector");
  hoverSizeSelector.style.display = 'flex';
}

// Hides the Hover box size choice div
function removeHoverSize () {
    // Hide size of Hover Square
    const hoverSizeSelector = document.querySelector("#hoverSizeSelector");
    hoverSizeSelector.style.display = 'none';
}

// Shows the Areas that display what is in the hover box and their resutls
function showCanvas () {
  const allCanvasArea = document.querySelector('.allCanvas');
  allCanvasArea.style.display = 'flex';
}

// Hides the Areas that display what is in the hover box and their resutls
function removeCanvas () {
  const allCanvasArea = document.querySelector('.allCanvas');
  allCanvasArea.style.display = 'none';
}


function showHoverSquare () {
  const allHoverSquare = document.querySelector('.hoverSquare');
  allHoverSquare.style.display = 'flex'; 
}

function removeHoverSquare () {
  const allHoverSquare = document.querySelector('.hoverSquare');
  allHoverSquare.style.display = 'none';
  allHoverSquare.style.width = '100px';
  allHoverSquare.style.height = '100px';

  const boxWidthInput = document.getElementById('boxWidth');
  boxWidthInput.value = '';
  const boxHeightInput = document.getElementById('boxHeight');
  boxHeightInput.value = '';
}

function removeImageResize() {
  const imageResizeSelector = document.querySelector("#imageResize");
  imageResizeSelector.style.display = 'none';
}

function showImageResize() {
  const imageResizeSelector = document.querySelector("#imageResize");
  imageResizeSelector.style.display = 'flex';
  imageResizeSelector.style.flexDirection = "column";
}

function removeSelectImagePrompt () {
  const selectedImagePrompt = document.querySelector("#clickImagePrompt");
  selectedImagePrompt.style.display = 'none';
}

function showSelectImagePrompt () {
  const selectedImagePrompt = document.querySelector("#clickImagePrompt");
  selectedImagePrompt.style.display = 'flex';
}

function showRotateAngle () {
  const selectedRotateElement = document.querySelector("#rotateAngle");
  selectedRotateElement.style.display = 'flex';
}

function removeRotateAngle () {
  const selectedRotateElement = document.querySelector("#rotateAngle");
  selectedRotateElement.style.display = 'none';
  rotateImageBool = false;
}

function showSwapColour () {
  const selectedRotateElement = document.querySelector("#colourSelection");
  selectedRotateElement.style.display = 'flex';
}

function removeSwapColour () {
  const selectedRotateElement = document.querySelector("#colourSelection");
  selectedRotateElement.style.display = 'none';
  rotateImageBool = true;
}

function colourChoice () {
  var form = document.getElementById('colorForm');
  
  // Loop through radio buttons to find the selected one
  var selectedOption;
  for (var i = 0; i < form.elements.length; i++) {
    if (form.elements[i].type === 'radio' && form.elements[i].checked) {
      colorChoiceVar = form.elements[i].value;
      break;
    }
  }
}

function removeSimpleThresh () {
  const selectedSimpleThreshElement = document.querySelector("#simpleThresh");
  selectedSimpleThreshElement.style.display = 'none';
}

function showSimpleThresh () {
  const selectedSimpleThreshElement = document.querySelector("#simpleThresh");
  selectedSimpleThreshElement.style.display = 'flex';
}

function removeThreshVals () {
  const selectedThreshElement = document.querySelector("#threshVals");
  selectedThreshElement.style.display = 'none';
}

function showHistPlot () {
  const selectedHistElement = document.querySelector("#plotlyDiv");
  selectedHistElement.style.display = 'block';
}

function removeHistPlot () {
  const selectedHistElement = document.querySelector("#plotlyDiv");
  selectedHistElement.style.display = 'none';
}

function showTranslate () {
  const selectedTranslateElement = document.querySelector("#translateDistanceSelector");
  selectedTranslateElement.style.display = 'flex';
}

function removeTransalte () {
  const selectedTranslateElement = document.querySelector("#translateDistanceSelector");
  selectedTranslateElement.style.display = 'none';
}

function showAffineTransform () {
  const selectedAffineTranformElement = document.querySelector("#affineTransfromSelector");
  selectedAffineTranformElement.style.display = 'flex';
}

function removeAffineTransform () {
  const selectedAffineTranformElement = document.querySelector("#affineTransfromSelector");
  selectedAffineTranformElement.style.display = 'none';
}

function updateTranslateDist() {
  // Assuming you want to update the main element with the "hoverSquare" class
  const translateXDist = document.getElementById("translateX").value;
  const translateYDist = document.getElementById("translateY").value;

  translateDistances = [translateXDist,translateYDist]
}

function updateAffineTransform () {
  const affineAngle = document.getElementById("affineAngle").value;
  const affineScale = document.getElementById("affineScale").value;

  affineTransformChoices = [affineAngle,affineScale]
}

function showThreshVals () {
  const selectedThreshElement = document.querySelector("#threshVals");
  selectedThreshElement.style.display = 'flex';
}

function removeThreshVals () {
  const selectedThreshElement = document.querySelector("#threshVals");
  selectedThreshElement.style.display = 'none';
}

function showAdaptThreshVals () {
  const selectedThreshElement = document.querySelector("#adaptiveThresh");
  selectedThreshElement.style.display = 'flex';
}

function removeAdaptThreshVals () {
  const selectedThreshElement = document.querySelector("#adaptiveThresh");
  selectedThreshElement.style.display = 'none';
}

function simpleThreshChoice() {
  // Get all radio buttons with the name "thresholdMethod"
  var radioButtons = document.getElementsByName("thresholdMethod");

  // Iterate through the radio buttons to find the selected one
  for (var i = 0; i < radioButtons.length; i++) {
    if (radioButtons[i].checked) {
      // The value attribute contains the selected checkbox value
      selectedSimpleThresholdMethod = radioButtons[i].value;
    }
  }
}

function updateThresholdVals() {
  // Get the values from the input fields
  var newThresholdValue = document.getElementById("thresholdValue").value;
  var newThresholdMax = document.getElementById("thresholdMax").value;

  // Check if a value was entered for thresholdValue and update if it exists
  if (newThresholdValue !== "") {
    thresholdValue = newThresholdValue;
  } else {
    thresholdValue = 127;
  }

  // Check if a value was entered for thresholdMax and update if it exists
  if (newThresholdMax !== "") {
    thresholdMax = newThresholdMax;
  } else {
    thresholdMax = 255;
  }
}

function adaptiveThreshChoice() {
  const adaptiveMaxValue = document.getElementById('maxValue').value;
  const adaptiveMethod = document.getElementById('adaptiveMethod').value;
  const adaptiveThresholdType = document.getElementById('thresholdType').value;
  const adaptiveBlockSize = document.getElementById('blockSize').value;
  const adaptiveConstant = document.getElementById('constantC').value;

  adaptiveParamaters = [adaptiveMaxValue, adaptiveMethod, adaptiveThresholdType, adaptiveBlockSize , adaptiveConstant]
}


function uploadRemoves () {
  resetMainDrop();
  resetAllSecondDrops();
  removeSecondaryChoiceDivs();
  removeAreaChoice();
  removeHoverSize();
  removeCanvas();
  removeHoverSquare();
  removeImageResize();
  removeSelectImagePrompt();
  removeSwapColour();
  removeRotateAngle();
  removeSimpleThresh();
  removeHistPlot();
  removeTransalte();
  removeAffineTransform(); 
  removeThreshVals();
  removeAdaptThreshVals();
}

function mainDropDownRemoves () {
  resetAllSecondDrops();
  removeSecondaryChoiceDivs();
  removeAreaChoice();
  removeHoverSize();
  removeImageResize();
  removeSelectImagePrompt();
  removeSwapColour();
  removeRotateAngle();
  removeSimpleThresh();
  removeThreshVals();
  removeHistPlot();
  removeTransalte();
  removeAffineTransform();
  removeThreshVals();
  removeAdaptThreshVals();

}

function secondaryDropDownRemoves () {
  removeAreaChoice();
  removeHoverSquare();
  removeImageResize();
  removeSelectImagePrompt();
  removeHoverSize();
  removeRotateAngle();
  removeSwapColour();
  removeSimpleThresh();
  removeThreshVals();
  removeHistPlot();
  removeTransalte();
  removeAffineTransform();
  removeThreshVals();
  removeAdaptThreshVals();
}

function createPlotlyHistogram (histdata) {
  
  // Break the data up into colours
  const histDataPlotly = [
    { y: histdata.histogramVals[0], type: 'lines', name: 'Red', line: { color: 'red' } },
    { y: histdata.histogramVals[1], type: 'lines', name: 'Green', line: { color: 'green' } },
    { y: histdata.histogramVals[2], type: 'lines', name: 'Blue', line: { color: 'blue' } }
  ];  

  // Create the layout
  const layout = {
    title: 'RGB Histogram',
    xaxis: { title: 'RGB Value' },
    yaxis: { title: 'Frequency' }
  };
  showHistPlot ();
  // Plot the plot
  Plotly.newPlot('plotlyDiv', histDataPlotly, layout);
}

// Functions to upload a new file
document.getElementById('imageUpload').addEventListener('change', function (event) {
  // Remove & Reset all previous Activity
  uploadRemoves(); 

  // Get the uploaded file
  var selectedFile = event.target.files[0];
  var reader = new FileReader();

  reader.onload = function (e) {
      var img = new Image();
      img.onload = function () {
          // Set the maximum width and height for the resized image
          var maxWidth = 800;
          var maxHeight = 600;

          // Calculate the aspect ratio of the image
          aspectRatio = img.width / img.height;
          usedAspectRatio = true;
          // Calculate new dimensions while maintaining aspect ratio
          var newWidth = Math.min(img.width, maxWidth);
          var newHeight = newWidth / aspectRatio;

          // Check if the new height exceeds the maxHeight
          if (newHeight > maxHeight) {
              newHeight = maxHeight;
              newWidth = newHeight * aspectRatio;
          }

          // Create a canvas element
          var canvas = document.createElement('canvas');
          var ctx = canvas.getContext('2d');

          // Set the canvas dimensions to the new dimensions
          canvas.width = newWidth;
          canvas.height = newHeight;

          // Draw the image onto the canvas with the new dimensions
          ctx.drawImage(img, 0, 0, newWidth, newHeight);

          // Get the data URL of the resized image
          var resizedImageDataUrl = canvas.toDataURL('image/jpeg');

          // Update the source (src) of the image element with the resized image
          document.getElementById('mainImage').src = resizedImageDataUrl;

          // Additional actions after setting the resized image
          setNewInitialImage();
      };

      // Set the source of the image to the data URL
      img.src = e.target.result;
  };

// Read the selected file as a data URL
reader.readAsDataURL(selectedFile);

});

// JavaScript: Updated showSecondDropdown function
function showSecondDropdown() {
  mainDropDownRemoves ();

  // Get the selected value from the main dropdown
  const selectedCategory = document.getElementById("mainDropdown").value;

  // Show the selected sub-dropdown
  if (selectedCategory !== "mainCategory") {
    const selectedSubForm = document.getElementById(selectedCategory);
    if (selectedSubForm) {
      selectedSubForm.style.display = "flex";
      selectedSubForm.style.flexDirection = "column";
      selectedSubForm.style.margin = "5px";
    }
  }
}

// show area options
function showSecondDropChoice(dropdownId) {
  secondaryDropDownRemoves();

  // Get the selected value from the main dropdown
  secondDropDownChoice = document.querySelector(`#${dropdownId}`).value;
  
  let entireList = ["resize","translate"];
  let selectedList = ["crop"];
  let choiceList = ["grayscale","rotate","swapColour","swapColour","simpleThresh","adaptThresh","otsuThresh","imageHist","histEqua","affine"]; 
           
  if (choiceList.includes(secondDropDownChoice)) {         
    console.log('CHOICE LIST')
    removeImageResize();
    removeCanvas();
    showAreaChoice();
    
    if (secondDropDownChoice == 'rotate') {
      showRotateAngle()
      } else if (secondDropDownChoice == 'swapColour') {
        // currentColourScheme = 'rgbColour' // THIS LINE COULD BE PROBLEMATIC
        showSwapColour()
      } else if (secondDropDownChoice == 'simpleThresh') {
        // currentColourScheme = 'rgbColour' // THIS LINE COULD BE PROBLEMATIC
        showSimpleThresh()
        showThreshVals ()
      } else if (secondDropDownChoice == 'affine') {
        showAffineTransform ()
      } else if (secondDropDownChoice == 'adaptThresh') {
        showAdaptThreshVals ()
      }

      
      
  } else if (selectedList.includes(secondDropDownChoice))  {
    removeAreaChoice();
    showHoverSize ();
    showCanvas();
    showHoverSquare();
    selectedArea = true;
    
  } else if (entireList.includes(secondDropDownChoice))  {
    removeCanvas();
    if (secondDropDownChoice == 'translate') {
      showTranslate ()
    } else if (secondDropDownChoice == "resize") {
      showImageResize()
    }
    removeAreaChoice();
   
    
    selectedArea = false; 
    }
} 

// function that does the choice of the area radio buttons
function areaChoice() {
  // Get all radio buttons with the name "radioSelection"
  removeSelectImagePrompt();
  removeHistPlot();
  const radioButtons = document.getElementsByName("areaSelectionInput");
  selectedAreaType = true
  // Loop through the radio buttons to find the selected one
  for (const radio of radioButtons) {
    if (radio.checked && radio.value === 'selectedArea') {
      // The selected radio button is found
      selectedArea = true;
      const hoverSizeSelector = document.querySelector("#hoverSizeSelector");
      hoverSizeSelector.style.display = 'flex';
      break;
    } else {
      selectedArea = false;
      // Hide size of Hover Square
      const hoverSizeSelector = document.querySelector("#hoverSizeSelector");
      hoverSizeSelector.style.display = 'none';    
    }
    showSelectImagePrompt();
  }  

  // Update the display based on the selectedArea value
  const allCanvasArea = document.querySelector('.allCanvas');
  const allHoverSquare = document.querySelector('.hoverSquare');
  
  if (selectedArea) {
    allCanvasArea.style.display = 'flex';
    allHoverSquare.style.display = 'flex';
  } else {
    allCanvasArea.style.display = 'none';
    allHoverSquare.style.display = 'none';
  }
}

// Function to get the relative x and y position of cursor on image
function imageRelative(e) {
  const hoverSquare = document.querySelector('.hoverSquare');
  const imageContainer = document.querySelector('.image-container');

  // Use clientX and clientY relative to the viewport
  const mouseX = e.clientX;
  const mouseY = e.clientY;

  // Adjust the square's position based on its dimensions
  let temp_left = mouseX - hoverSquare.offsetWidth;
  let temp_top = mouseY - hoverSquare.offsetHeight;

  // Update the square's position considering the scroll position
  temp_left += window.scrollX || document.documentElement.scrollLeft;
  temp_top += window.scrollY || document.documentElement.scrollTop;

  hoverSquare.style.left = temp_left + 'px';
  hoverSquare.style.top = temp_top + 'px';

  const containerRect = imageContainer.getBoundingClientRect();
  const relativeX = mouseX - containerRect.left;
  const relativeY = mouseY - containerRect.top;

  return [relativeX, relativeY];
}

// Shows what is in the blue square
document.querySelector('.image-container').addEventListener('mousemove', function (e) {
    let relativeXY = imageRelative(e);
    let relativeX = relativeXY[0];
    let relativeY = relativeXY[1];

    // Update the size of the window to show what is in the sqyare
    const hoverElement = document.querySelector(".hoverSquare");   
    let snippetWidth, snippetHeight;

    if (isNaN(parseInt(hoverElement.style.width))) {
      snippetWidth = 100;
      snippetHeight = 100;
      hoverElement.style.width = snippetWidth
      hoverElement.style.height = snippetHeight
    } else {
      snippetWidth = parseInt(hoverElement.style.width);
      snippetHeight = parseInt(hoverElement.style.height);
    }    

    // Shows what is in the square on mousemove
    let selectedImageInfo = showSnippet(relativeX-snippetWidth, relativeY-snippetHeight, snippetWidth, snippetHeight);

    imageData = selectedImageInfo[0];
    imageDataHeight = selectedImageInfo[1];
    imageDataWidth = selectedImageInfo[2];
  });

// Function to show what is in the moving square
function showSnippet(leftVal, topVal, sourceWidth, sourceHeight) {
  const canvas = document.getElementById("myCanvasFollow");
  const ctx = canvas.getContext("2d");
  const img = document.getElementById("mainImage");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw the image onto the temporary canvas with the correct coordinates
  ctx.drawImage(img, leftVal,topVal,sourceWidth,sourceHeight,0,0,canvas.width,canvas.height
    );

  // Get the image data from the temporary canvas as an ImageData object
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  return [imageData, canvas.width, canvas.height];
}

function updateImageSize () {
  selectedImageWidth = document.getElementById("imageSelectedBoxWidth").value;
  selectedImageHeight = document.getElementById("imageSelectedBoxHeight").value;

  showSelectImagePrompt();
}

function rotateAngle () {
  selectedRotateAngle = document.getElementById("rotateAngleSelection").value;
}

// Event to display square when in image container
document.querySelector('.image-container').addEventListener('mouseenter', function () {
  if (selectedArea == true) {
    document.querySelector('.hoverSquare').style.display = 'flex';
  }
});

// Event to remove square when out image container
document.querySelector('.image-container').addEventListener('mouseleave', function () {
  document.querySelector('.hoverSquare').style.display = 'none';
  
});

// initiates sending the image to python
document.querySelector('.image-container').addEventListener('click', function (e) {
  // Paste the Selected image after image effect applied
  
    if (!selectedArea) {
      // Get the image element by ID
      const mainImageElement = document.getElementById("mainImage");

      // Create an offscreen canvas
      const offscreenCanvas = document.createElement("canvas");
      const offscreenContext = offscreenCanvas.getContext("2d");

      // Set the canvas size to match the image size
      offscreenCanvas.width = mainImageElement.width;
      offscreenCanvas.height = mainImageElement.height;

      // Draw the image onto the offscreen canvas
      offscreenContext.drawImage(mainImageElement, 0, 0);

      // Get the image data from the offscreen canvas as an ImageData object
      imageData = offscreenContext.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
      if (!savedInitialImage) {
        setNewInitialImage()
        }    
      imageDataHeight = mainImageElement.width;
      imageDataWidth =  mainImageElement.height;
      }
    
    sendImageSnippet(imageData,imageDataHeight,imageDataWidth,secondDropDownChoice);
  
});

// Resets the main image to the origional image
// Updates the size of the hover box
function updateHoverBox() {
  const hoverElement = document.querySelector(".hoverSquare");

  // Assuming you want to update the main element with the "hoverSquare" class
  const boxWidth = document.getElementById("boxWidth").value;
  const boxHeight = document.getElementById("boxHeight").value;

  // Show the hover box if selectedArea is true
  if (selectedArea) {
    hoverElement.style.display = 'flex';
  }

  if (!usedAspectRatio) {
    hoverElement.style.display = 'flex';
    hoverElement.style.width = boxWidth + 'px';
    hoverElement.style.height = boxHeight + 'px';
  } else {
    hoverElement.style.display = 'flex';
    hoverElement.style.width = boxWidth/aspectRatio  + 'px';
    hoverElement.style.height =  boxHeight/aspectRatio + 'px';
  }
  showSelectImagePrompt();
}

// Resets the main image to the origional image
function resetInitialImage() {
  const imgElement = document.getElementById("mainImage");
  imgElement.src = initialImage;
  removeHistPlot();
}

// Function to send and retrieve image to show
function sendImageSnippet(clickedImage,clickedImageHeight,clickedImageWidth,selectedImageProcess ) {

  // Send the image data to the server using a fetch request
  fetch('/process_image', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ imageData: clickedImage,
                            imageHeight:clickedImageHeight,
                            imageWidth: clickedImageWidth,
                            imageProcess: selectedImageProcess,
                            imageWidthSelected: selectedImageWidth,
                            imageHeightSelected: selectedImageHeight,
                            imageTranslateDistances: translateDistances,
                            imageAffineTransform: affineTransformChoices,
                            imageRotateAngle: selectedRotateAngle,
                            imageCurrentColourScheme : currentColourScheme,
                            imageColourChoice: colorChoiceVar,
                            imageselectedSimpleThreshold : selectedSimpleThresholdMethod,
                            imagethresholdValue: thresholdValue,
                            imageAdaptiveParamaters : adaptiveParamaters, 
                            imagethresholdMax: thresholdMax 
                          }),
    })

  .then(response => response.json())
  .then(data => {
    // Update Current Colour Scheme
    tempCurrentColourScheme = data.currentColourScheme;
    if (tempCurrentColourScheme !== null) {
      currentColourScheme = tempCurrentColourScheme
    }

    console.log('data', data)
    console.log('data.histogramVals', data.histogramVals)
    
    // Deal with Histogram of the image
    if (data.histogramVals && data.histogramVals.length > 0) {
      createPlotlyHistogram(data);
    }

    if (!selectedArea) {
      // Get the image element by ID
      const imgElement = document.getElementById("mainImage");
      imgElement.src = 'data:image/png;base64,' + data.img;
    } else {
      const canvas = document.getElementById('myCanvasEffect');
      const ctx = canvas.getContext('2d');
      // Create an Image object
      const img = new Image();
      // Set the image source to the base64-encoded image data from the JSON response
      img.src = 'data:image/png;base64,' + data.img;

      // After the image is loaded, draw it on the canvas
      img.onload = function() {
          ctx.drawImage(img, 0, 0);
        };
      }    
    })
  .catch(error => {
    console.error('Error processing image:', error);
    });
  };