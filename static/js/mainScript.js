// Set Variables
let selectedArea1 = false;
let selectedArea2 = false;
let entireImage = true;
let placeImageCanvas = false;
let imageData;
let imageDataHeight;
let imageDataWidth;
let secondDropDownChoice = "";
let savedInitialImage = false;
// Initialize the initial image from the HTML
let initialImage = document.getElementById("sourceImage").src;
let initialImageWidth = document.getElementById("sourceImage").naturalWidth;
let initialImageHeight = document.getElementById("sourceImage").naturalHeight;

let usedAspectRatio = false;
let aspectRatio;
let selectedImageWidth; 
let selectedImageHeight;
let selectedRotateAngle = 45;
let currentColourScheme = 'rgbColour';
let desiredColorScheme = 'rgbColour';
let tempCurrentColourScheme;
let selectedSimpleThresholdMethod;
let thresholdValue = 127;
let thresholdMax = 255;
let translateDistances = [50,30];
let affineTransformChoices = [45,1.5];
let adaptiveParamaters = []
let selectedKernel;
let morphSelection;
let contourFeatureSelection;
let contourBoundingBoxSelection;
let fftFilterSelection;
let selectedEdgeDetection;
let clusterSeg;
let binClassModel;
let entireImageData;
let objDetModel;
let clickedimageProcess;
let clickedBinModel;
let mainImageElement;
let mainImageCanvas;
let clickedClassModel;
let sliderChoice;


function isPositiveInteger(value) {
  const num = Number(value);
  return Number.isInteger(num) && num > 0;
}

function isInteger(value) {
  const num = Number(value);
  return Number.isInteger(num);
}

function isFloat(value) {
  // Convert to number and check if it's a finite float
  const num = Number(value);
  return !isNaN(num) && num % 1 !== 0 && isFinite(num);
}

function isNumber(value) {
  // Check if the value is a number
  return !isNaN(parseFloat(value)) && isFinite(value);
}

//
document.addEventListener('click', e => {
  const isDropDownButton = e.target.matches("[data-dropdown-button]");
  const isMainOption = e.target.matches(".main-option");
  const isSubOption = e.target.matches(".sub-option");

  if (isMainOption) {
      e.preventDefault();
      // Update the subtitle
      const mainDropdown = document.querySelector('.mainForm .dropdown');
      const label = e.target.getAttribute("data-label");
      mainDropdown.querySelector(".subtitle").textContent = label;

      // Hide all subForms
      document.querySelectorAll(".subForm").forEach(subForm => {
          subForm.classList.remove("active");
      });

      // Show selected subForm
      const value = e.target.getAttribute("data-value");
      document.querySelector(`.subForm.${value}`).classList.add("active");

      // Keep the dropdown open
      const dropdown = e.target.closest("[data-dropdown]");
      dropdown.classList.add("active");
  }

  if (isSubOption) {
      e.preventDefault();
      // Update the sub subtitle
      const subDropdown = e.target.closest(".subForm .dropdown");
      const label = e.target.getAttribute("data-label");
      subDropdown.querySelector(".sub-subtitle").textContent = label;
      const dataVal = e.target.getAttribute("data-value");
      console.log('DataVal',dataVal);  
      showSecondDropChoice(dataVal);
      
  }
  // Close open dropdowns when clicking outside any dropdown.
  if (!isDropDownButton && e.target.closest('[data-dropdown]') == null) {
      document.querySelectorAll("[data-dropdown].active").forEach(dropdown => {
          dropdown.classList.remove("active");
      });
      return;
  }
  // Toggle the clicked dropdown when clicking on a dropdown button.
  let currentDropdown;
  if (isDropDownButton) {
      currentDropdown = e.target.closest("[data-dropdown]");
      currentDropdown.classList.toggle("active");
  }
  // Close other open dropdowns when toggling a specific dropdown.
  document.querySelectorAll("[data-dropdown].active").forEach(dropdown => {
      if (dropdown === currentDropdown) return;
      dropdown.classList.remove("active");
  });
});


// Updates the Main Image to return to when Reset Chosen
function setNewInitialImage(imageData, width, height) {
  console.log('Setting new initial image');
  initialImage = imageData;
  initialImageWidth = width;
  initialImageHeight = height;
}

// Updates the main image with a user selected image
function getImage(useUploaded, event, callback) {
  // Default to a saved image
  var selectedFile = null;

  if (useUploaded) {
      // Get the uploaded file
      selectedFile = event.target.files[0];
  } else {
      // Provide a saved image path or URL
      selectedFile = initialImage; // Replace this with the actual saved image path or URL
  }

  var img = new Image();
  var reader = new FileReader();

  img.onload = function () {
      // Set the maximum width and height for the resized image
      var maxWidth = 500;
      var maxHeight = 500;

      // Original dimensions of the image
      var originalWidth = img.width;
      var originalHeight = img.height;

      // Determine new dimensions
      var newWidth = originalWidth;
      var newHeight = originalHeight;
      var aspectRatio = originalWidth / originalHeight;

      // Resize only if the image exceeds the max dimensions
      if (originalWidth > maxWidth || originalHeight > maxHeight) {
          if (originalWidth > originalHeight) {
              newWidth = maxWidth;
              newHeight = newWidth / aspectRatio;
          } else {
              newHeight = maxHeight;
              newWidth = newHeight / aspectRatio;
          }
      }

      // Create a canvas element
      var canvas = document.getElementById('imageCanvas');
      var ctx = canvas.getContext('2d');

      // Set the canvas dimensions to the new dimensions
      canvas.width = newWidth;
      canvas.height = newHeight;

      // Draw the image onto the canvas with the new dimensions
      ctx.drawImage(img, 0, 0, newWidth, newHeight);

      // Get the data URL of the resized image
      var resizedImageDataUrl = canvas.toDataURL('image/jpeg');

      // Update the source (src) of the image element with the resized image
      document.getElementById('sourceImage').src = resizedImageDataUrl;

      // Set the new initial image and its dimensions
      setNewInitialImage(resizedImageDataUrl, newWidth, newHeight);

      // Call the callback function, if provided, indicating image processing is complete
      if (callback && typeof callback === 'function') {
          callback(canvas);
      }
  };

  if (useUploaded) {
      reader.onload = function (e) {
          // Set the source of the image to the data URL
          img.src = e.target.result;
      };
      // Read the selected file as a data URL
      reader.readAsDataURL(selectedFile);
  } else {
      // Set the source of the image directly
      img.src = selectedFile;
  }
}

// Resets the main image to the original image
function resetInitialImage() {
  console.log('ResetImage');
  // Get the source image element and canvas
  var mainImageElement = document.getElementById("sourceImage");
  var canvas = document.getElementById("imageCanvas");
  var ctx = canvas.getContext("2d");

  // Restore the original image to the image element
  mainImageElement.src = initialImage;

  // Create a new Image object
  var imgElement = new Image();
  imgElement.onload = function () {
      // Once the image is loaded, adjust the canvas size
      canvas.width = initialImageWidth;
      canvas.height = initialImageHeight;

      // Draw the image onto the canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas first
      ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);
  };
  imgElement.src = initialImage; // Set the source of the Image object

  // Remove hover squares and class predictions
  removeAllCanvas();
}
  
// Function to upload a new file
document.getElementById('imageUpload').addEventListener('change', function (event) {
  // Remove & Reset all previous Activity
  uploadRemoves(); 
  getImage(true, event, function() { });
});

// Functions to upload a new file for pred
document.getElementById('predImageUploadForm').addEventListener('change', function (event) {
  const fieldset = document.getElementById('predUploadField');
  fieldset.style.borderColor = 'green';
  // Set useUploaded to true since an image is being uploaded
  const useUploaded = true;  
  // Call getImage and provide a callback to sendPredImage
  getImage(useUploaded, event, function() {
    sendPredImage(clickedimageProcess);
  });
  
});

// Function to get information about the image
function getImageParams() {
  // Get the image element by ID
  mainImageElement = document.getElementById('sourceImage');

  // Create an offscreen canvas
  const offscreenCanvas = document.createElement("canvas");
  const offscreenContext = offscreenCanvas.getContext("2d");

  // Set the canvas size to match the image size
  offscreenCanvas.width = mainImageElement.naturalWidth;
  offscreenCanvas.height = mainImageElement.naturalHeight

  // Draw the image onto the offscreen canvas
  offscreenContext.drawImage(mainImageElement, 0, 0);

  // Get the image data from the offscreen canvas as an ImageData object
  let imageDataTemp = offscreenContext.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);


  return imageDataTemp;
  };

function getImageDataUrl() {
  const mainImageElement = document.getElementById("imageCanvas");
  const offscreenCanvas = document.createElement("canvas");
  const offscreenContext = offscreenCanvas.getContext("2d");

  offscreenCanvas.width = mainImageElement.naturalWidth;
  offscreenCanvas.height = mainImageElement.naturalHeight;

  offscreenContext.drawImage(mainImageElement, 0, 0);

  // Get the data URL of the image from the canvas
  const imageDataUrl = offscreenCanvas.toDataURL('image/jpeg');
  // if (!savedInitialImage) {
  //   setNewInitialImage()
  //   }    

  return imageDataUrl;
}

// Load the initial image
window.onload = function () {
  var useUploaded = false; // Set this based on whether an uploaded or saved image should be used.
  getImage(useUploaded, null, function (canvas) {
      initializeCanvas(canvas);
  });
};

// Function that allows for functionilaty on the main image
function initializeCanvas(canvas) {
    var context = canvas.getContext('2d');
    var img = document.getElementById('sourceImage'); // Ensure the img element is in your HTML.

    var isDragging = false;
    var startPoint = { x: 0, y: 0 };
    var endPoint = { x: 0, y: 0 };

    canvas.onmousedown = function(event) {
        if (!selectedArea2) return; // Allow dragging only if selectedArea2 is true.
        isDragging = true;
        startPoint = getMousePos(canvas, event);
    };

    canvas.onmousemove = function(event) {
        if (!selectedArea2 || !isDragging) return; // Continue only if selectedArea2 is true and the user is dragging
        endPoint = getMousePos(canvas, event);
    
        // Calculate the width and height of the rectangle
        var width = endPoint.x - startPoint.x;
        var height = endPoint.y - startPoint.y;
    
        // Clear the canvas and redraw the image
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
    
        // Draw the selection rectangle
        context.beginPath();
        context.rect(startPoint.x, startPoint.y, width, height);
        context.strokeStyle = 'red';
        context.stroke();
    
        // Set style for the dimension text
        context.fillStyle = 'blue';
        context.font = '14px Arial';
        
        // Calculate the positions for the dimension text
        var textWidthPosition = startPoint.x + (width / 2) - (context.measureText("Width: " + Math.abs(width) + "px").width / 2);
    
        // Display the width of the rectangle at the top
        context.fillText("Width: " + Math.abs(width) + "px", textWidthPosition, startPoint.y - 5);
        
        // For height, display it to the side. Adjust the X position slightly to the left or right of the rectangle
        // Depending on whether the rectangle is being drawn to the left or right, adjust the placement
        var textHeightXPosition = startPoint.x + width + 5; // Adjust this value as needed
        // Ensure the height text doesn't overlap with the rectangle by checking the direction
        if (width < 0) {
            textHeightXPosition = startPoint.x - context.measureText("Height: " + Math.abs(height) + "px").width - 10; // Move text to the left of the start point
        }
        
        var textHeightYPosition = startPoint.y + height / 2 + 4; // Centered along the height, adjust as needed
    
        // Display the height of the rectangle horizontally along the side
        context.fillText("Height: " + Math.abs(height) + "px", textHeightXPosition, textHeightYPosition);
    };    

    function saveRectangleArea(img, x1, y1, x2, y2) {
        // Calculate the dimensions of the rectangle
        var width = Math.abs(x2 - x1);
        var height = Math.abs(y2 - y1);
    
        // Create a new canvas element to draw the selected area
        var saveCanvas = document.createElement('canvas');
        saveCanvas.width = width;
        saveCanvas.height = height;
        var saveCtx = saveCanvas.getContext('2d');
    
        // Draw the selected area of the original image onto the new canvas
        saveCtx.drawImage(img, x1, y1, width, height, 0, 0, width, height);
        var imageData = saveCtx.getImageData(0, 0, width, height);
    
        // Convert the canvas to a data URL and create a link to download it
        // var dataUrl = saveCanvas.toDataURL('image/png');
        sendImageSnippet(imageData.data,saveCanvas.height,saveCanvas.width,secondDropDownChoice);
    }
    
    // Modify the canvas.onmouseup event handler to call saveRectangleArea
    canvas.onmouseup = function(event) {
        if (!selectedArea2) return; // Only proceed if selectedArea2 is true
        isDragging = false;
        endPoint = getMousePos(canvas, event);
    
        var x1 = Math.min(startPoint.x, endPoint.x);
        var y1 = Math.min(startPoint.y, endPoint.y);
        var x2 = Math.max(startPoint.x, endPoint.x);
        var y2 = Math.max(startPoint.y, endPoint.y);
    
        // Call saveRectangleArea to save the selected part of the image
        saveRectangleArea(img, x1, y1, x2, y2);
    
        // Optionally, clear the selection rectangle after saving
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    
}

function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
}

// Resets the Main Drop Down Menu
function resetMainDrop () {
  // Reset the dropdowns to the initial default value
  const mainDropdownElement = document.getElementById("mainDropdown");
  mainDropdownElement.selectedIndex = 0;
}

// Resets the Secondary Drop Down Menu
function resetAllSecondDrops() {
  const secondDropdowns = document.querySelectorAll(".secondForm");

  secondDropdowns.forEach(dropdown => {
    dropdown.selectedIndex = 0;
  });
}

// Function to automatically check the appropriate radio button
// function selectedAreaRadio() {
//   // Access the radio buttons by their IDs
//   var selectedAreaRadioButton = document.getElementById('selectedArea');
//   var entireImageRadioButton = document.getElementById('entireImage');

//   if (selectedArea) {
//     // If isSelectedArea is true, select the 'Selected Area' radio button
//     selectedAreaRadioButton.checked = true;
//   } else {
//     // If isSelectedArea is false, select the 'Entire Image' radio button
//     entireImageRadioButton.checked = true;
//   }
// }

// Shows the Area Selected Div
function showAreaChoice () {
  //Hide area Elements
  const selectedAreaType = document.getElementById("areaSelection"); 
  selectedAreaType.style.display = "flex";
  selectedAreaType.style.flexDirection = "column";
  selectedAreaType.style.justifyContent = "centre";
  selectedAreaType.style.margin = "5px";
}

//// SHOW REMOVE ////
// Hides the Areas that display what is in the hover box and their resutls
function showSlider(newLabelText) {
  const allSliderElem = document.getElementById('numberSlider');
  const sliderInput = allSliderElem.querySelector('.slider'); // Get the existing slider input
  const currentSliderValue = allSliderElem.querySelector('#sliderValue').textContent; // Get the current slider value
  
  // Update the innerHTML to include both the new label text and preserve the slider input
  allSliderElem.innerHTML = `${newLabelText}: <span id="sliderValue">${currentSliderValue}</span>`;
  allSliderElem.insertBefore(sliderInput, allSliderElem.firstChild); // Re-insert the slider input at the beginning

  // Re-apply styles and make it visible
  allSliderElem.style.display = 'flex';

  // Reattach the event listener to update the display value
  sliderInput.oninput = function() {
    document.getElementById('sliderValue').textContent = this.value;
    sliderChoice = document.getElementById('sliderValue').textContent
  };
}

function removeSlider() {
  const allSliderElem = document.getElementById('numberSlider');
  allSliderElem.style.display = 'none'; 
}

function removeCanvas () {
  const allCanvasArea = document.querySelector('.allCanvas');
  allCanvasArea.style.display = 'none';
}

function showHoverSquare () {
  const allHoverSquare = document.querySelector('.hoverSquare');
  allHoverSquare.style.display = 'flex'; 
}

function removeHoverSquare() {
  const allHoverSquares = document.querySelector('.hoverSquare');
  allHoverSquares.style.display = 'none';
  };

function showHoverSize () {
  // Shows size of Hover Square
  const hoverSizeSelector = document.querySelector("#hoverSizeSelector");
  hoverSizeSelector.style.display = 'flex';
}

function showImageResize() {
  const imageResizeSelector = document.querySelector("#imageResize");
  imageResizeSelector.style.display = 'flex';
  imageResizeSelector.style.flexDirection = "column";
}

function showSelectImagePrompt () {
  const selectedImagePrompt = document.querySelector("#clickImagePrompt");
  selectedImagePrompt.style.display = 'flex';
}

function showRotateAngle () {
  const selectedRotateElement = document.querySelector("#rotateAngle");
  selectedRotateElement.style.display = 'flex';
}

function showSwapColour () {
  const selectedRotateElement = document.querySelector("#colourSelection");
  selectedRotateElement.style.display = 'flex';
}

function showSimpleThresh () {
  const selectedSimpleThreshElement = document.querySelector("#simpleThresh");
  selectedSimpleThreshElement.style.display = 'flex';
}

function showTranslate () {
  const selectedTranslateElement = document.querySelector("#translateDistanceSelector");
  selectedTranslateElement.style.display = 'flex';
}

function showThreshVals () {
  const selectedThreshElement = document.querySelector("#threshVals");
  selectedThreshElement.style.display = 'flex';
}

function showAffineTransform () {
  const selectedAffineTranformElement = document.querySelector("#affineTransfromSelector");
  selectedAffineTranformElement.style.display = 'flex';
}

function showSmoothingKernel () {
  const selectedThreshElement = document.querySelector("#smoothingKernel");
  selectedThreshElement.style.display = 'flex';
}

function showEdgeKernel () {
  const selectedThreshElement = document.querySelector("#edgeKernel");
  selectedThreshElement.style.display = 'flex';
}

function showCustomKernel () {
  const selectedThreshElement = document.querySelector("#customeKernelButton");
  selectedThreshElement.style.display = 'block';
}

function showMorphologicalKernel () {
  const selectedThreshElement = document.querySelector("#morphologicalKernel");
  selectedThreshElement.style.display = 'flex';
}

function showShapeChoice () {
  const selectedThreshElement = document.querySelector("#drawShapeSelection");
  selectedThreshElement.style.display = 'flex';
}

function showFftFilter () {
  const selectedFftFilterElement = document.querySelector("#FftFilter");
  selectedFftFilterElement.style.display = 'flex';
}

function showSegTable() {
  const selectedSegTableElement = document.querySelector("#segTable");
  selectedSegTableElement.style.display = 'flex';
}

function showClusterSeg () {
  const selectedFeaturehElement = document.querySelector("#clusterSeg");
  selectedFeaturehElement.style.display = 'flex';
}

function showBinClass () {
  const selectedBinClassElement = document.querySelector("#binModelSelection");
  selectedBinClassElement.style.display = 'flex';
}

function showMultiClass () {
  const selectedBinClassElement = document.querySelector("#multiModelSelection");
  selectedBinClassElement.style.display = 'flex';
}

function showClassButton() {
  document.querySelector('#predImageUploadForm').parentNode.style.display = 'flex';
}

function showBoundingBoxChoice () {
  const selectedFeaturehElement = document.querySelector("#contourBoundingBox");
  selectedFeaturehElement.style.display = 'flex';
}

function showEdgeDetectionChoice () {
  const selectedFeaturehElement = document.querySelector("#edgeDetection");
  selectedFeaturehElement.style.display = 'flex';
}

function showcontourFeatureChoice () {
  const selectedFeaturehElement = document.querySelector("#contourFeatureSelection");
  selectedFeaturehElement.style.display = 'flex';
}


////////////////////

function updateTranslateDist() {
  const fieldset = document.getElementById('translateDistanceSelector');
  const translateXDist = document.getElementById("translateX").value;
  const translateYDist = document.getElementById("translateY").value;

  // Check if both inputs are positive integers
  if (isInteger(translateXDist) && isPositiveInteger(translateYDist)) {
    // Change the border color to green
    fieldset.style.borderColor = 'green';
    translateDistances = [translateXDist,translateYDist]
  } else {
      // Change the border color to red and show an alert
      fieldset.style.borderColor = 'red';
      alert('Please enter two positive integers for X and Y.');
  }   
}

function updateAffineTransform () {
  const fieldset = document.getElementById('affineTransfromSelector');
  const affineAngle = document.getElementById("affineAngle").value;
  const affineScale = document.getElementById("affineScale").value;

  // Check if both inputs are positive integers
  if (isNumber(affineAngle) && isInteger(affineScale)) {
    // Change the border color to green
    fieldset.style.borderColor = 'green';
    affineTransformChoices = [affineAngle,affineScale]
  } else {
      // Change the border color to red and show an alert
      fieldset.style.borderColor = 'red';
      alert('Please enter two integers for X and Y.');
  }  

  
}


////////////////////////

function showCanvas () {
  const allCanvasArea = document.querySelector('.allCanvas');
  allCanvasArea.style.display = 'grid';
}

function showCanvasFollow () {
  const selectedCanvasElement = document.querySelector('#myCanvasFollow');
  selectedCanvasElement.style.display = 'block';
}

function removeCanvasFollow () {
  const selectedCanvasElement = document.querySelector('#myCanvasFollow');
  selectedCanvasElement.style.display = 'none';
}

function showMainCanvas1 () {
  const selectedCanvasElement = document.querySelector('#mainImageCanvas1');
  selectedCanvasElement.style.display = 'block';
}

function removeMainCanvas1 () {
  const selectedCanvasElement = document.querySelector('#mainImageCanvas1');
  selectedCanvasElement.style.display = 'none';
}

function showMainCanvas2 () {
  const selectedCanvasElement = document.querySelector('#mainImageCanvas2');
  selectedCanvasElement.style.display = 'block';
  updateCanvasGrid();
}

function removeMainCanvas2 () {
  const selectedCanvasElement = document.querySelector('#mainImageCanvas2');
  selectedCanvasElement.style.display = 'none';
}

function showMainCanvas3 () {
  const selectedCanvasElement = document.querySelector('#mainImageCanvas3');
  selectedCanvasElement.style.display = 'block';
  updateCanvasGrid();
}

function removeMainCanvas3 () {
  const selectedCanvasElement = document.querySelector('#mainImageCanvas3');
  selectedCanvasElement.style.display = 'none';
}


function showSubCanvas1 () {
  const selectedCanvasElement = document.querySelector('#subCanvas1');
  selectedCanvasElement.style.display = 'block';
}

function removeSubCanvas1 () {
  const selectedCanvasElement = document.querySelector('#subCanvas1');
  selectedCanvasElement.style.display = 'none';
}

function showSubCanvas2 () {
  const selectedCanvasElement = document.querySelector('#subCanvas2');
  selectedCanvasElement.style.display = 'block';
  updateCanvasGrid();
}

function removeSubCanvas2 () {
  const selectedCanvasElement = document.querySelector('#subCanvas2');
  selectedCanvasElement.style.display = 'none';
}

function showSubCanvas3 () {
  const selectedCanvasElement = document.querySelector('#subCanvas3');
  selectedCanvasElement.style.display = 'block';
  updateCanvasGrid();
}

function removeSubCanvas3 () {
  const selectedCanvasElement = document.querySelector('#subCanvas3');
  selectedCanvasElement.style.display = 'none';
}


function showCanvasDiv1 () {
  const selectedCanvasElement = document.querySelector('#myCanvasDiv1');
  selectedCanvasElement.style.display = 'block';
  updateCanvasGrid();
}

function removeCanvasDiv1 () {
  const selectedCanvasElement = document.querySelector('#myCanvasDiv1');
  selectedCanvasElement.style.display = 'none';
}

function showCanvasDiv2 () {
  const selectedCanvasElement = document.querySelector('#myCanvasDiv2');
  selectedCanvasElement.style.display = 'block';
  updateCanvasGrid();
}

function removeCanvasDiv2 () {
  const selectedCanvasElement = document.querySelector('#myCanvasDiv2');
  selectedCanvasElement.style.display = 'none';
}

function showCanvasDiv3 () {
  const selectedCanvasElement = document.querySelector('#myCanvasDiv3');
  selectedCanvasElement.style.display = 'block';
  updateCanvasGrid();
}

function removeCanvasDiv3 () {
  const selectedCanvasElement = document.querySelector('#myCanvasDiv3');
  selectedCanvasElement.style.display = 'none';
}

function showResetCanvasButton1 () {
  const selectedCanvasElement = document.querySelector('#resetCanvasButton1');
  selectedCanvasElement.style.display = 'block';
  updateCanvasGrid();
}

function removeResetCanvasButton1 () {
  const selectedCanvasElement = document.querySelector('#resetCanvasButton1');
  selectedCanvasElement.style.display = 'none';
}

function showResetCanvasButton2 () {
  const selectedCanvasElement = document.querySelector('#resetCanvasButton2');
  selectedCanvasElement.style.display = 'block';
  updateCanvasGrid();
}

function removeResetCanvasButton2 () {
  const selectedCanvasElement = document.querySelector('#resetCanvasButton2');
  selectedCanvasElement.style.display = 'none';
}

function showResetCanvasButton3 () {
  const selectedCanvasElement = document.querySelector('#resetCanvasButton3');
  selectedCanvasElement.style.display = 'block';
  updateCanvasGrid();
}

function removeResetCanvasButton3 () {
  const selectedCanvasElement = document.querySelector('#resetCanvasButton3');
  selectedCanvasElement.style.display = 'none';
}

function showAdaptThreshVals () {
  const selectedThreshElement = document.querySelector("#adaptiveThresh");
  selectedThreshElement.style.display = 'flex';
}


function showNextFreeCanvas (nextFreeCanvasId) {
  if (nextFreeCanvasId == 'mainImageCanvas1') {
    showMainCanvas1()
    showResetCanvasButton1()
  } else if (nextFreeCanvasId == 'mainImageCanvas2') {
    showMainCanvas2()
    showResetCanvasButton2()
  } else if (nextFreeCanvasId == 'mainImageCanvas3') {
    showMainCanvas3()
    showResetCanvasButton3()
  } else if (nextFreeCanvasId == 'subCanvas1') {
    showSubCanvas1 ()
  } else if (nextFreeCanvasId == 'subCanvas2') {
    showSubCanvas2 ()
  } else if (nextFreeCanvasId == 'subCanvas3') {
    showSubCanvas3 ()
  } else if (nextFreeCanvasId == 'myCanvasDiv1') {
    showCanvasDiv1()
  } else if (nextFreeCanvasId == 'myCanvasDiv2') {
    showCanvasDiv2()
  } else if (nextFreeCanvasId == 'myCanvasDiv3') {
    showCanvasDiv3()
  }
} 

function resetIndCanvas (canvasId) {
  if (canvasId == 'mainImageCanvas1') {
    removeMainCanvas1()
    removeCanvasDiv1()
    removeSubCanvas1 ()
    removeResetCanvasButton1()
  } else if (canvasId == 'mainImageCanvas2') {
    removeMainCanvas2 ()
    removeCanvasDiv2()
    removeSubCanvas2 ()
    removeResetCanvasButton2()
  } else if (canvasId == 'mainImageCanvas3') {
    removeMainCanvas3 ()
    removeCanvasDiv3()
    removeSubCanvas3 ()
    removeResetCanvasButton3()
  } 
}

function removeAllCanvas () {
  removeCanvasFollow();
  removeMainCanvas1();
  removeMainCanvas2();
  removeMainCanvas3();
  removeCanvasDiv1();
  removeCanvasDiv2();
  removeCanvasDiv3();
  removeResetCanvasButton1();
  removeResetCanvasButton2();
  removeResetCanvasButton3();
}

////////////////////////


//////////////////
function colourChoice() {
  desiredColorScheme = document.querySelector('input[name="colourSelection"]:checked').value;
}

function morphChoice() {
  const fieldset = document.getElementById('morphologicalKernel');
  // Get the selected radio button value
  morphSelection = document.querySelector('input[name="morphSelection"]:checked').value;
  fieldset.style.borderColor = 'green';
}

function contourFeatureChoice() {
  const fieldset = document.getElementById('contourFeatureSelection');
  // Get the selected radio button value
  contourFeatureSelection = document.querySelector('input[name="contourFeatureSelectionInput"]:checked').value;
  fieldset.style.borderColor = 'green';
}

function edgeDetectionChoice() {
  // Get the selected radio button value
  selectedEdgeDetection = document.querySelector('input[name="edgeDetectionSelection"]:checked').value;
  secondDropDownChoice = 'edgeDetection'
  removeImageResize();
  showAreaChoice();
  showCanvas ();
  // Check if Selected Area Type is Ticked 
  if (selectedArea1 || selectedArea2) {
    showCanvasFollow ();
    showHoverSquare();
    placeImageCanvas = true;
        } 
}

function contourBoundingBoxChoice() {
  const fieldset = document.getElementById('contourBoundingBox');
  // Get the selected radio button value
  contourBoundingBoxSelection = document.querySelector('input[name="contourBoundingBoxInput"]:checked').value;
  fieldset.style.borderColor = 'green';
}

function binClass() {
  const fieldset = document.getElementById('binModelSelection');
  clickedimageProcess = 'binaryClass'
  showClassButton();
  clickedBinModel = document.querySelector('input[name="binaryClassSelection"]:checked').value;
  fieldset.style.borderColor = 'green';
}

function multiClass() {
  clickedimageProcess = 'multiClass'
  showClassButton();
  clickedClassModel = document.querySelector('input[name="multiClassSelection"]:checked').value;
  const fieldset = document.getElementById('multiModelSelection');
  fieldset.style.borderColor = 'green';
}


function objectDetectionChoice() {
  clickedimageProcess = 'objectDetection'
  showClassButton();
  objDetModel = document.querySelector('input[name="objectDetectionModel"]:checked').value;
  
}

function clusterSegChoice() {
  const fieldset = document.getElementById('clusterSeg');
  clusterSeg = document.querySelector('input[name="clusterSegSelection"]:checked').value;
  if (clusterSeg == 'clusterKmeans') {
    showSlider('Number of Clusters')
  }
  if (clusterSeg == 'clusterMean') {
    removeSlider()
  }
  fieldset.style.borderColor = 'green';
}

function fftFilterChoice() {
  const fieldset = document.getElementById('FftFilter');
  fftFilterSelection = document.querySelector('input[name="selectedFftFilter"]:checked').value;
  fieldset.style.borderColor = 'green';
  
}

function getNumDisplayedCanvas(className) {
  // Get the parent element
  var canvasElements = document.getElementsByClassName(className);
  let totalDisplayed = 0;
  
  // Iterate through all children elements
  for (var i = 0; i < canvasElements.length; i++) {
    var computedStyle = window.getComputedStyle(canvasElements[i]);
    if (computedStyle.display !== 'none') {
      totalDisplayed += 1;
    }  
  }  
  return totalDisplayed;
}

function getNextFreeCanvas(className) {
  // Get the parent element
  var canvasElements = document.getElementsByClassName(className);
  let nextFreeCanvasId;

  // Iterate through all children elements
  for (var i = 0; i < canvasElements.length; i++) {
      var computedStyle = window.getComputedStyle(canvasElements[i]);
      if (computedStyle.display === 'none') {
          nextFreeCanvasId = canvasElements[i].id;
          break;
      }
  }

  return nextFreeCanvasId;
}

function updateCanvasGrid() {
  let totalDisplayed = getNumDisplayedCanvas();
  let numRows = Math.ceil(totalDisplayed / 2);

  const allCanvas = document.querySelector('.allCanvas');
  
  allCanvas.style.gridTemplateRows = `repeat(${numRows}, 1fr)`;
}

function simpleThreshChoice() {
  const fieldset = document.getElementById('simpleThresh');
  // Get all radio buttons with the name "thresholdMethod"
  selectedSimpleThresholdMethod = document.querySelector('input[name="thresholdMethod"]:checked').value;
  fieldset.style.borderColor = 'green'; 
  }


function updateThresholdVals() {
  const fieldset = document.getElementById('threshVals')
  // Get the values from the input fields
  var newThresholdValue = document.getElementById("thresholdValue").value;
  var newThresholdMax = document.getElementById("thresholdMax").value;

  
  // Check if both inputs are positive integers
  if (isPositiveInteger(newThresholdValue) && isPositiveInteger(newThresholdMax)) {
    // Change the border color to green
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
    fieldset.style.borderColor = 'green';
  } else {
    // Change the border color to red and show an alert
    fieldset.style.borderColor = 'red';
    alert('Please enter two positive integers for Threshold Value and Max.');
  }  
}

function adaptiveThreshChoice() {
  const fieldset = document.getElementById('adaptiveThresh')
  const adaptiveMaxValue = document.getElementById('maxValue').value;
  const adaptiveMethod = document.getElementById('adaptiveMethod').value;
  const adaptiveThresholdType = document.getElementById('thresholdType').value;
  const adaptiveBlockSize = document.getElementById('blockSize').value;
  const adaptiveConstant = document.getElementById('constantC').value;


  if (isPositiveInteger(adaptiveMaxValue) && isPositiveInteger(adaptiveBlockSize) && isInteger(adaptiveConstant) ) {
    adaptiveParamaters = [adaptiveMaxValue, adaptiveMethod, adaptiveThresholdType, adaptiveBlockSize , adaptiveConstant]
    fieldset.style.borderColor = 'green';
  } else {
    // Change the border color to red and show an alert
    fieldset.style.borderColor = 'red';
    alert('Please enter the Max Value and Block Size as a positive integers and C as an Integer');
  }   
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
    yaxis: { title: 'Frequency' },
    width: 300,  // Set the width of the plot
    height: 200 
  };

  let nextFreeCanvasId = getNextFreeCanvas('divCanvas');
  showNextFreeCanvas(nextFreeCanvasId);
  Plotly.newPlot(nextFreeCanvasId, histDataPlotly, layout);
}

function createFftThresh (data) {
  
  let nextFreeCanvasId = getNextFreeCanvas('subCanvas');
  showNextFreeCanvas(nextFreeCanvasId);
  drawImageInCanvas(data.fftThresh, nextFreeCanvasId)
}

function updateImageSize () {
  const fieldset = document.getElementById('imageResize');
  selectedImageWidth = document.getElementById("imageSelectedBoxWidth").value;
  selectedImageHeight = document.getElementById("imageSelectedBoxHeight").value;
  
  const height = selectedImageHeight;
  const width = selectedImageWidth;

  // Check if both inputs are not empty and are numbers
    // Check if both inputs are positive integers
  if (isPositiveInteger(width) && isPositiveInteger(height)) {
      // Change the border color to green
      fieldset.style.borderColor = 'green';
  } else {
      // Change the border color to red and show an alert
      fieldset.style.borderColor = 'red';
      alert('Please enter two positive integers for width and height.');
  }  
}

function rotateAngle() {
  const fieldset = document.getElementById('rotateAngle');
  selectedRotateAngle = document.getElementById("rotateAngleSelection").value;
  const rotateAngle = selectedRotateAngle
  
  if (isNumber(rotateAngle)) {
      // Change the border color to green
      fieldset.style.borderColor = 'green';
  } else {
      // Change the border color to red and show an alert
      fieldset.style.borderColor = 'red';
      alert('Please enter a valid float for the Rotation Angle.');
  }  
}

function smoothingKernelChoice() {
  const fieldset = document.getElementById('smoothingKernel');
  // Get the selected radio button value
  selectedKernel = document.querySelector('input[name="smoothingKernelSelection"]:checked').value;
  fieldset.style.borderColor = 'green';
}

function edgeKernelChoice() {
  const fieldset = document.getElementById('edgeKernel');
  // Get the selected radio button value
  selectedKernel = document.querySelector('input[name="edgeKernelSelection"]:checked').value;
  fieldset.style.borderColor = 'green';
}

// function that does the choice of the area radio buttons
function areaChoice() {

  // Get the selected radio button
  const selectedRadioButton = document.querySelector('input[name="areaSelectionInput"]:checked');

  if (selectedRadioButton.value === 'selectedArea1') {
      // The "Selected Area" radio button is selected
      selectedArea1 = true;
      selectedArea2 = false;
      entireImage = false;
      placeImageCanvas = true;
      showCanvasFollow();
      showHoverSize();
      showHoverSquare();
    } else if (selectedRadioButton.value === 'selectedArea2')  {
      removeHoverSquare();
      removeCanvasFollow();
      selectedArea1 = false;
      selectedArea2 = true;
      entireImage = false;
      placeImageCanvas = true;
    } else {
      // Another radio button is selected
      showSelectImagePrompt();
      removeHoverSquare();
      removeCanvasFollow();
      entireImage = true;
      selectedArea1 = false;
      selectedArea2 = false;
      placeImageCanvas = false;
    }
  } 

//////////////////////////////////
function drawImageInCanvas(dataImg, nextFreeCanvasId) {
  const canvas = document.getElementById(nextFreeCanvasId);
  const ctx = canvas.getContext('2d');
  // Create an Image object
  const img = new Image();
  // Set the image source to the base64-encoded image data from the JSON response
  img.src = 'data:image/jpeg;base64,' + dataImg;
  // After the image is loaded, draw it on the canvas
  img.onload = function() {
    canvas.width = img.width;
    canvas.height = img.height;  
    ctx.drawImage(img, 0, 0);
  }
}

function openCustomKernelWindow () {
  var newWindow = window.open("/kernel_popup", "NewWindow", "width=600, height=400, resizable=yes, scrollbars=yes");

  newWindow.receiveDataFromPopup = receiveDataFromPopup;
}

function receiveDataFromPopup(data) {
  // Handle the data received from the popup
  selectedKernel = data;
}
///////////////////////////////
function hideSecondForms() {
  // Get all elements with the class name 'secondForm'
  var elements = document.getElementsByClassName('secondForm');
  
  // Loop through the elements and set their display style to 'none'
  for(var i = 0; i < elements.length; i++) {
    elements[i].style.display = 'none';
  }
}

function hideSecondFormsParams() {
  // Get all elements with the class name 'secondForm'
  var elements = document.getElementsByClassName('secondFormParams');
  
  // Loop through the elements and set their display style to 'none'
  for(var i = 0; i < elements.length; i++) {
    elements[i].style.display = 'none';
  }
}

function uploadRemoves () {
  resetMainDrop();
  resetAllSecondDrops();
  hideSecondForms();
  hideSecondFormsParams();

  selectedArea1 = false;
}

function mainDropDownRemoves () {
  resetAllSecondDrops();
  hideSecondForms();
  hideSecondFormsParams();
  selectedArea1 = false;
}

function secondaryDropDownRemoves () {
  hideSecondFormsParams();
  selectedArea1 = false;
}

// Script that controls the slider

document.addEventListener('DOMContentLoaded', function () {
  const slider = document.querySelector('#numberSlider .slider');  // Correctly select the input range element
  const sliderOutput = document.getElementById('sliderValue');

  slider.oninput = function() {
    sliderOutput.textContent = this.value;
    sliderChoice =  sliderOutput.textContent
  };
});


/////////////////////////

// JavaScript: Updated showSecondDropdown function
// function showSecondDropdown() {
//   mainDropDownRemoves ();

//   // Get the selected value from the main dropdown
//   const selectedCategory = document.getElementById("mainDropdown").value;
//   // Show the selected sub-dropdown
//   if (selectedCategory !== "mainCategory") {
//     const selectedSubForm = document.getElementById(selectedCategory);
//     if (selectedSubForm) {
//       selectedSubForm.style.display = "flex";
//       selectedSubForm.style.flexDirection = "column";
//       selectedSubForm.style.margin = "5px";
//     }
//   }
// }

// show area options
function showSecondDropChoice(subChoice) {
  secondDropDownChoice = subChoice
  secondaryDropDownRemoves();
  console.log('second label', secondDropDownChoice);
 
  let entireList = ["resize","translate","FftSpectrum","FftFilter","clusterSeg","binaryClass","multiClass","threshSeg"];
  let selectedList = ["crop"];
  let choiceList = ["grayscale","rotate","swapColour","swapColour","simpleThresh","adaptThresh","otsuThresh","imageHist","histEqua","affine",
  "identityKernel","smoothingKernel","sharpeningKernel","edgeDetectionKernel","morphologicalKernel","frequencyDomainKernel","customKernel",
  "drawContours",'drawShapes',"contourFeatures","boundingFeatures","edgeDetection","segmentationOpsDropDown"]; 
            
  if (choiceList.includes(secondDropDownChoice)) {         
    showAreaChoice();
    showCanvas ();
    // Check if Selected Area Type is Ticked 
    if (selectedArea1 || selectedArea2) {
      showCanvasFollow ();
      showHoverSquare();
      placeImageCanvas = true;
          } 
    
    if (secondDropDownChoice == 'rotate') {
      showRotateAngle()
      } else if (secondDropDownChoice == 'swapColour') {
        showSwapColour()
      } else if (secondDropDownChoice == 'simpleThresh' ) {
        showSimpleThresh()
        showThreshVals ()
      } else if (secondDropDownChoice == 'affine') {
        showAffineTransform ()
      } else if (secondDropDownChoice == 'adaptThresh') {
        showAdaptThreshVals ()
      } else if (secondDropDownChoice == 'otsuThresh') {
        showThreshVals ()     
      } else if (secondDropDownChoice == 'smoothingKernel') {
        showSmoothingKernel ()     
      } else if (secondDropDownChoice == 'edgeDetectionKernel') {
        showEdgeKernel ()
      } else if (secondDropDownChoice == 'morphologicalKernel') {
        showMorphologicalKernel ()
      } else if (secondDropDownChoice == 'customKernel') {
        showCustomKernel ()
        // showStrideChoice ()
        // showPaddingChoice ()
      } else if (secondDropDownChoice == 'drawShapes') {
        showShapeChoice()
      } else if (secondDropDownChoice == 'contourFeatures') {
        showcontourFeatureChoice()
      } else if (secondDropDownChoice == 'boundingFeatures') {
        showBoundingBoxChoice()
      } else if (secondDropDownChoice == 'edgeDetection') {
        showEdgeDetectionChoice () 
      }                 

  } else if (selectedList.includes(secondDropDownChoice))  {
      selectedArea1 = true;
      showHoverSize ();
      showCanvas();
      showHoverSquare();
      placeImageCanvas = true;
    
    } else if (entireList.includes(secondDropDownChoice))  {
      selectedArea1 = false;
      selectedArea2 = false;
      entireImage = true;
      if (secondDropDownChoice == 'translate') {
        showTranslate ()
      } else if (secondDropDownChoice == "resize") {
        showImageResize()
      } else if (secondDropDownChoice == "FftSpectrum") {
        showSelectImagePrompt ()
        placeImageCanvas = true
      } else if (secondDropDownChoice == "FftFilter") {
        showFftFilter ()
        placeImageCanvas = true
      } else if (secondDropDownChoice == 'clusterSeg') {
        showClusterSeg();
      } else if (secondDropDownChoice == 'watershed') {
        showSelectImagePrompt();
      } else if (secondDropDownChoice == 'binaryClass') {
        showBinClass()
      } else if (secondDropDownChoice == 'multiClass') {
        showMultiClass()
      } else if (secondDropDownChoice == 'threshSeg') {
        showSlider('Number of Thresholds')
      }       
    }
} 

// Function to get the relative x and y position of cursor on image
function imageRelative(e) {
  const hoverSquare = document.querySelector('.hoverSquare');
  const imageContainer = document.querySelector('#imageCanvas');

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
document.querySelector('#imageCanvas').addEventListener('mousemove', function (e) {
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
  const canvasTrial = document.getElementById("myCanvasFollow");

  canvasTrial.width = sourceWidth;
  canvasTrial.height = sourceHeight;

  const ctx = canvasTrial.getContext("2d");
  const img = document.getElementById("sourceImage");
  ctx.clearRect(0, 0, canvasTrial.width, canvasTrial.height);

  // Draw the image onto the temporary canvas with the correct coordinates
  // ctx.drawImage(img, leftVal,topVal,sourceWidth,sourceHeight,0,0,canvas.width,canvas.height);
  ctx.drawImage(img, leftVal,topVal,sourceWidth,sourceHeight,0,0,sourceWidth,sourceHeight);

  // Get the image data from the temporary canvas as an ImageData object
  const imageData = ctx.getImageData(0, 0, canvasTrial.width, canvasTrial.height);
  return [imageData, canvasTrial.width, canvasTrial.height];
}

// Event to display square when in image container
document.querySelector('#imageCanvas').addEventListener('mouseenter', function () {
  console.log('selectedArea1',selectedArea1)
  if (selectedArea1) {
      document.querySelector('.hoverSquare').style.display = 'flex';
      }
  });

// Event to remove square when out image container
document.querySelector('#imageCanvas').addEventListener('mouseleave', function () {
  document.querySelector('.hoverSquare').style.display = 'none';

});

// initiates sending the image to python
document.querySelector('#imageCanvas').addEventListener('click', function (e) {
    
    if (selectedArea1) {
      imageData = imageData.data;
    }

    if (selectedArea2) {
      return;
    }    
    
    if (entireImage) {
      entireImageData = getImageParams();
      imageData = entireImageData.data;
      imageDataHeight = entireImageData.height;
      imageDataWidth = entireImageData.width;
    }
    // Get a snippet sized image

    sendImageSnippet(imageData,imageDataHeight,imageDataWidth,secondDropDownChoice);
  
});

// Updates the size of the hover box
function updateHoverBox() {
  const hoverElement = document.querySelector(".hoverSquare");
  const fieldset = document.getElementById('hoverSizeSelector');
  // Assuming you want to update the main element with the "hoverSquare" class
  const boxWidth = document.getElementById("boxWidth").value;
  const boxHeight = document.getElementById("boxHeight").value;

  // Check if both inputs are positive integers
  if (isPositiveInteger(boxWidth) && isPositiveInteger(boxHeight)) {
    // Change the border color to green
    fieldset.style.borderColor = 'green';
    // Show the hover box if selectedArea is true
    if (selectedArea1 || selectedArea2) {
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
    } else {
        // Change the border color to red and show an alert
        fieldset.style.borderColor = 'red';
        alert('Please enter two positive integers for width and height.');
    } 
}




function jsonReplaceMainImg(data) {
  console.log('json.replace.main.image')
  // Create an Image object
  const imgElement = new Image();
  imgElement.src = 'data:image/jpeg;base64,' + data.img;

  // Wait for the image to load
  imgElement.onload = function() {
    // Get the canvas element
    const canvas = document.getElementById("imageCanvas");
    const ctx = canvas.getContext('2d');

    // Adjust the canvas size to the image size
    canvas.width = imgElement.naturalWidth;
    canvas.height = imgElement.naturalHeight;

    // Clear the canvas and draw the new image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

  };
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
                            imageDesiredColorScheme: desiredColorScheme,
                            imageselectedSimpleThreshold : selectedSimpleThresholdMethod,
                            imagethresholdValue: thresholdValue,
                            imageAdaptiveParamaters : adaptiveParamaters, 
                            imagethresholdMax: thresholdMax,
                            imageselectedKernel : selectedKernel,
                            imageMorphSelection : morphSelection,
                            imageContourFeatureSelection : contourFeatureSelection,
                            imageContourBoundingBoxSelection : contourBoundingBoxSelection,
                            imagefftFilterSelection : fftFilterSelection,
                            imageSelectedEdgeDetection : selectedEdgeDetection,
                            imageClusterSeg : clusterSeg,
                            imagesliderOutput : sliderChoice  
                          }),
    })

  .then(response => response.json())
  .then(data => {
    // Update Current Colour Scheme
    tempCurrentColourScheme = data.currentColourScheme;
    if (tempCurrentColourScheme !== null) {
      currentColourScheme = tempCurrentColourScheme
    }    
    // Deal with Histogram of the image
    if (data.histogramVals && data.histogramVals.length > 0) {
      createPlotlyHistogram(data);
    }
    
    if (data.fftThresh && data.fftThresh.length > 0) {
      createFftThresh(data);
    }
    if (data.semanticBool == true) {
      initializeDataTable();
    }

    if (!placeImageCanvas) {
      jsonReplaceMainImg(data)
    } else {
      // Add the image to the smaller canvases
      let nextFreeCanvasId = getNextFreeCanvas('mainCanvas');
      showNextFreeCanvas(nextFreeCanvasId);
      drawImageInCanvas(data.img, nextFreeCanvasId)
      }

    })
  .catch(error => {
    console.error('Error processing image:', error);
    });
  };

function swapClassPredsToText (binPreds,multiPreds) {
  let classText = '';

  // Check if binPreds is not False (i.e., it is a valid number)
  if (binPreds !== false) {
    if (binPreds > 0.5) {
      // If binPreds is greater than 0.5, classify as dog
      classText = `The Image is of a dog with ${binPreds} probability`;
    } else {
      // Otherwise, calculate the probability for cat and classify as cat
      let catProb = 1 - binPreds;
      classText = `The Image is of a cat with ${catProb} probability`;
    }
  } else {
    // If binPreds is False, process multiple predictions
    for (let i = 0; i < multiPreds.length; i++) {
      let classPredI = multiPreds[i][1]; // Assuming the class name is at index 1
      let classProbI = multiPreds[i][2]; // Assuming the probability is at index 2
      let tempText = `The class probabilities are ${classPredI} with ${classProbI} probability `;
      classText += tempText;
    }
  }
  
  return classText;
}

function updateClassPredText (newText) {
    // Select the paragraph element by its ID
    var paragraph = document.getElementById('classPreds');

    // Update the text content of the paragraph
    paragraph.textContent = newText;
    paragraph.style.display = 'flex';
}

function showClassPreds(binPreds,multiPreds) {
  classText = swapClassPredsToText (binPreds,multiPreds)
  updateClassPredText (classText)
}  

// Function to make predictions on image
function sendPredImage(clickedimageProcess) {
  const mainImageElement = document.getElementById('sourceImage');
  mainImageElement.onload = function() {
    // Call getImageParams only after the image has fully loaded
    let predEntireImageData = getImageParams();
    // Send the image data to the server using a fetch request
    fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ predImageData: predEntireImageData.data,
                              predImageHeight:predEntireImageData.height,
                              predImageWidth: predEntireImageData.width,
                              imageProcess: clickedimageProcess,
                              binModel : clickedBinModel,
                              multiModel : clickedClassModel,
                              detectionModel: objDetModel                           
                            }),
                          })
    .then(response => response.json())
    .then(data => {
      binPred = data.binPred;
      multiPred = data.multiPred; 
        
      showClassPreds(binPred,multiPred);     
        
        jsonReplaceMainImg(data)
      })
    .catch(error => {
      console.error('Error processing image:', error);
      });
    };
  };  

///////
function initializeDataTable() {
  showSegTable();

  var table = $('#dataTable').DataTable({
      ajax: {
          url: '/imgSegTable',
          dataSrc: ''
      },
      columns: [
          { title: "Row Num", data: "Row Num" },
          { title: "Class", data: "Classes" },
          { title: "Probabilities", data: "Probabilities" }
      ],
      select: true,
      initComplete: function() {
          updateClassCheckboxes();
      }
  });

  function updateClassCheckboxes() {
    var uniqueClasses = [];
    $('#classFilter').find('input[type="checkbox"]:not(#checkAllClasses)').remove();
    $('#classFilter').find('label:not([for="checkAllClasses"])').remove();

    table.column(1).data().each(function(value, index) {
        if (uniqueClasses.indexOf(value) === -1) {
            uniqueClasses.push(value);
            var checkboxId = 'class' + index;
            $('#classFilter').append(
                $('<input>').prop({
                    type: 'checkbox',
                    id: checkboxId,
                    name: 'class',
                    value: value,
                    checked: true,
                    class: 'class-checkbox'
                }),
                $('<label>').prop({
                    for: checkboxId
                }).text(value)
            );
        }
    });

    $('.class-checkbox').on('change', function() {
        // Check if all checkboxes are checked or not
        var allChecked = $('.class-checkbox').length === $('.class-checkbox:checked').length;
        $('#checkAllClasses').prop('checked', allChecked);
        performFilter();
    });

    $('#checkAllClasses').change(function() {
        // Apply the checked status of the 'All' checkbox to all class checkboxes
        var isChecked = $(this).is(':checked');
        $('.class-checkbox').prop('checked', isChecked);
        performFilter();
    });
}


  function performFilter() {
    var searchStr = $('.class-checkbox:checked').map(function() {
        return this.value;
    }).get().join('|');
    table.column(1).search(searchStr, true, false).draw();
  }

  $('#applyProbFilter').click(function() {
    var probValue = parseFloat($('#minProbability').val());
    table.column(2).search(probValue ? probValue.toString() : '', true, false).draw();
});


  // Toggle all checkboxes for segmentation options
  $('#segAllCheck').change(function() {
      var isChecked = $(this).is(':checked');
      $('#segBbCheck, #segOutlinesCheck, #segMasksCheck, #segCutCheck').prop('checked', isChecked);
  });

  // Uncheck 'Select All' if any individual checkbox is unchecked
  $('#segBbCheck, #segOutlinesCheck, #segMasksCheck, #segCutCheck').change(function() {
      if (!this.checked) {
          $('#segAllCheck').prop('checked', false);
      } else {
          if ($('#segBbCheck').is(':checked') && $('#segOutlinesCheck').is(':checked') && $('#segMasksCheck').is(':checked') && $('#segCutCheck').is(':checked')) {
              $('#segAllCheck').prop('checked', true);
          }
      }
  });

  // Handle segmentation processing
  $('#processBtn').click(function() {
      var selectedRowsData = table.rows({ selected: true, search: 'applied' }).data();
      var selectedRowNums = selectedRowsData.map(function (data) {
          return data['Row Num'];
      }).toArray();

      var segmentationOptions = {
          segBbCheck: $('#segBbCheck').is(':checked'),
          segOutlinesCheck: $('#segOutlinesCheck').is(':checked'),
          segMasksCheck: $('#segMasksCheck').is(':checked'),
          segCutCheck: $('#segCutCheck').is(':checked')
      };

      $.ajax({
          type: "POST",
          url: "/processSeg",
          data: JSON.stringify({ rowNumbers: selectedRowNums, options: segmentationOptions }),
          contentType: "application/json",
          success: function(response) {
              jsonReplaceMainImg(response);
          },
          error: function(err) {
              console.error('Error processing: ' + err);
          }
      });
  });
}


