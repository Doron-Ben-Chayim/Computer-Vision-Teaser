// Set Variables
let selectedAreaStamp 
let selectedAreaDrag 
let Imageentire = true;
let placeImageCanvas = false;
let imageData;
let imageDataHeight;
let imageDataWidth;
var secondDropDownChoice;
let savedInitialImage = false;
let initialImage = document.getElementById("sourceImage").src;
let initialImageWidth = document.getElementById("sourceImage").naturalWidth;
let initialImageHeight = document.getElementById("sourceImage").naturalHeight;
let usedAspectRatio = false;
let aspectRatio;
let selectedImageWidth; 
let selectedImageHeight;
let selectedRotateAngle = 45;
let tempCurrentColourScheme;
let selectedSimpleThresholdMethod;
let thresholdValue = 127;
let thresholdMax = 255;
let translateDistances = [50,30];
let translateXDist = 50;
let translateYDist = 30;
let affineAngle = 45;
let affineScale = 1.5;
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
let sliderChoice = 5;
let boxWidth = 100;
let boxHeight = 100;
let newThresholdValue = 127;
let newThresholdMax = 255;
let adaptiveMaxValue = 255;
let adaptiveMethod = 'meanAdapt';
let adaptiveThresholdType = 'gaussAdapt'
let adaptiveBlockSize = 3;
let adaptiveConstant = 0;
let turnImgBoundGreenList = ['Show Spectrum','Semantic']
let currentColorSchemeMain = 'rgbColour';
let desiredColorScheme = '';
let fileType = '';
let textOCRLst = [];
let currentTextOCRIndex = 0;
let ocrimgLst = [];
let imageDataTempOCR = '';
let fourrSliderChoice = '';
let fourrMaxCutoff = '';
let processingLoaderContainer = document.getElementById('processing-loader-container');
let processingLoader = document.querySelector('.processing-loader');
const fieldsets = document.querySelectorAll('fieldset.secondFormParams');
const mainImage = document.querySelector('#imageCanvas');
// MutationObserver to watch for changes in the fieldsets
const observer = new MutationObserver(checkIfReadyToClick);
const config = { attributes: true, childList: true, subtree: true };

///// CHECKING FUNCTIONS
function isPositiveInteger(value) {
  const num = Number(value);
  return Number.isInteger(num) && num > 0;
}

function isInteger(value) {
  const num = Number(value);
  return Number.isInteger(num);
}

function isOddInteger(value) {
  return Number.isInteger(value) && value % 2 !== 0;
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

function openReadMe() {
  window.open('/readme', '_blank');
}

function maxCutoffFrequency(initialImageHeight, initialImageWidth) {
  let halfHeight = initialImageHeight / 2;
  let halfWidth = initialImageWidth / 2;
  let fourrMaxCutoff = Math.sqrt(halfHeight**2 + halfWidth**2);
  return fourrMaxCutoff
}

///// OCR FUNCTIONS

function updateTextOCR() {
  const displayedText = document.getElementById("text-container"); // Select the <p> tag inside #text-container
  displayedText.textContent = textOCRLst[currentTextOCRIndex];
  showTessText()
}

function updateImageOCR(currentimgOCRIndex=currentTextOCRIndex) {
  const displayedImage = document.getElementById("ocrImage"); // Select the <img> element

  // Update the src attribute of the img element
  displayedImage.src = 'data:image/jpeg;base64,' + ocrimgLst[currentimgOCRIndex];
  showOCRImage()
}

function prevTextOCR() {
  if (currentTextOCRIndex > 0) {
      currentTextOCRIndex--;
  } else {
      currentTextOCRIndex = textOCRLst.length - 1; // Wrap around to the last item
  }
  updateTextOCR();
  updateImageOCR();
  displayPageCount(currentTextOCRIndex+1, textOCRLst.length)
}

function nextTextOCR() {
  if (currentTextOCRIndex < textOCRLst.length - 1) {
      currentTextOCRIndex++;
  } else {
      currentTextOCRIndex = 0; // Wrap around to the first item
  }
  updateTextOCR();
  updateImageOCR();
  displayPageCount(currentTextOCRIndex+1, textOCRLst.length)
}

function ocrPageSelection(numberOfBoxes) {
  const checkboxList = document.getElementById('checkbox-list');

  // Clear existing checkboxes
  checkboxList.innerHTML = '';

  // Add new checkboxes
  for (let i = 1; i <= numberOfBoxes; i++) {
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.id = `checkbox-${i}`;
      checkbox.name = `checkbox-${i}`;
      checkbox.className = 'text-checkbox';
      checkbox.value = i;

      label.appendChild(checkbox);
      label.appendChild(document.createTextNode(` Image ${i}`));
      checkboxList.appendChild(label);
  }
}

function toggleSelectAll(selectAllCheckbox) {
  const checkboxes = document.querySelectorAll('.text-checkbox');
  checkboxes.forEach(checkbox => {
      checkbox.checked = selectAllCheckbox.checked;
  });
}

function askChatGPT() {  
  const selectedCheckboxes = document.querySelectorAll('.text-checkbox:checked');
  const question = document.getElementById('question').value;
  const chatAPI = document.getElementById('ChatKey').value;
  let selectedText = [];

  selectedCheckboxes.forEach(checkbox => {
    let index = parseInt(checkbox.value) - 1;  
    selectedText.push(textOCRLst[index]);
  });
  if (selectedText.length === 0) {
      alert("Please select at least one text.");
      return;
  }

  const requestData = {
      text: selectedText.join(" "),
      question: question,
      chatAPI: chatAPI
  };

  showLoading();
  

  fetch('/ask-chatgpt', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById('response').textContent = data.chatGPTResponse;
    removeLoading();
  })
  .catch(error => {
      console.error('Error:', error);
  });
}

///// USER ACCESIBILITY FUNCTIONS

function downloadImage(id) {
  const canvas = document.getElementById(id);
  const image = canvas.toDataURL('image/png');
  const link = document.createElement('a');
  link.href = image;
  link.download = 'canvas_image.png';
  link.click();
}

function getVisibleSubSubtitleTextLength() {
  const subSubtitles = document.querySelectorAll('.sub-subtitle');
  for (let subSubtitle of subSubtitles) {
      if (subSubtitle.offsetParent !== null) {
          return subSubtitle.textContent.length;
      }
  }
  return 0;
}

function getVisibleSubSubtitleText() {
  const subSubtitles = document.querySelectorAll('.sub-subtitle');
  for (let subSubtitle of subSubtitles) {
      if (subSubtitle.offsetParent !== null) {
          return subSubtitle.textContent;
      }
  }
}

// Checks to see if the kernel has swapped to green
function isKernelElementVisibleAndGreen() {
  const element = document.querySelector('#customeKernelButton p');
  
  // Check visibility
  const isVisible = element.offsetParent !== null;
  
  // Get computed style of the element
  const style = window.getComputedStyle(element);
  const isGreen = style.color === 'rgb(0, 128, 0)'; // RGB for green

  if (!isVisible) {
    return false;}

  if (isVisible && !isGreen) {
    return false;}

  if (isVisible && isGreen) {
      return true;
  }
}

// Function to check if all visible fieldsets have a green border and other conditions
function checkIfReadyToClick() {
  let allVisibleGreenFieldset = true;

  fieldsets.forEach((fieldset) => {
      const computedStyle = window.getComputedStyle(fieldset);
      if (computedStyle.display !== 'none' && computedStyle.borderColor !== 'rgb(0, 128, 0)') {
        allVisibleGreenFieldset = false;
      }
  });

  const visibleElementTextLength = getVisibleSubSubtitleTextLength();
  const visibleElementText = getVisibleSubSubtitleText();
  
  let isSubtitlePresent = turnImgBoundGreenList.includes(visibleElementText);

  const led = document.getElementById('led');

  if (allVisibleGreenFieldset && visibleElementTextLength != 0) {
      led.classList.add('ready');
  } else if (isSubtitlePresent) {
      led.classList.add('ready');
  } else if (isKernelElementVisibleAndGreen()) {
      led.classList.add('ready');
  } else {
      led.classList.remove('ready');
  }
}

function turnLedReady() {
  led.classList.add('ready');
}


fieldsets.forEach((fieldset) => {
  observer.observe(fieldset, config);
});

checkIfReadyToClick();

document.addEventListener('DOMContentLoaded', function () {
  const segTable = document.getElementById('segTable');
  const toggleButton = document.getElementById('toggleButton');
  
  // Ensure segTable and toggleButton exist
  if (segTable && toggleButton) {
      // Initially hide the segTable
      segTable.style.display = 'none';

      // Function to toggle visibility
      function toggleSegTable() {
          if (segTable.style.display === 'none') {
              segTable.style.display = 'block';
          } else {
              segTable.style.display = 'none';
          }
      }

      // Add event listener to the button to toggle segTable
      toggleButton.addEventListener('click', toggleSegTable);
    }
});

function initializeToggleSwitches() {
  const toggleSwitchEntire = document.getElementById('toggleSwitchEntire');
  const toggleSwitchSnip = document.getElementById('toggleSwitchSnip');

  if (secondDropDownChoice === 'crop') {
      toggleSwitchEntire.disabled = true;
      toggleSwitchEntire.parentElement.style.opacity = 0.5; // Grays out the toggle switch
      toggleSwitchEntire.parentElement.querySelector('.slider').style.backgroundColor = 'gray';
      toggleSwitchSnip.parentElement.classList.remove('disabled');
      toggleSwitchSnip.disabled = false;
      toggleSwitchSnip.checked = false;
      areaChoice('selectedAreaDrag');
  } else {
      toggleSwitchEntire.disabled = false;
      toggleSwitchEntire.checked = true;
      toggleSwitchEntire.parentElement.classList.remove('disabled');
      toggleSwitchEntire.parentElement.style.opacity = 1; // unGrays out the toggle switch
      toggleSwitchEntire.parentElement.querySelector('.slider').style.backgroundColor = 'green';
      toggleSwitchSnip.disabled = true;
      toggleSwitchSnip.parentElement.classList.remove('disabled');
      toggleSwitchSnip.checked = false;
      areaChoice('selectedEntire');
  }

  if (toggleSwitchEntire) {
      toggleSwitchEntire.addEventListener('change', function() {
          const isCheckedEntire = toggleSwitchEntire.checked;

          if (toggleSwitchSnip) {
              toggleSwitchSnip.disabled = isCheckedEntire;
              areaChoice('selectedEntire');

              if (isCheckedEntire) {
                  toggleSwitchEntire.parentElement.querySelector('.slider').style.backgroundColor = 'green';
                  toggleSwitchSnip.parentElement.style.opacity = 0.5;
              } else {
                  toggleSwitchEntire.parentElement.querySelector('.slider').style.backgroundColor = 'gray';
                  toggleSwitchSnip.parentElement.style.opacity = 1;
                  selectedAreaDrag = true;
                  placeImageCanvas = true;
              }
          }
      });
  }

  if (toggleSwitchSnip) {
      entireArea = false;

      toggleSwitchSnip.addEventListener('change', function() {
          const isCheckedSnip = toggleSwitchSnip.checked;

          if (isCheckedSnip) {
              areaChoice('selectedAreaStamp');
              showHoverBox ();
              showHoverSquare();
              showCanvas();
          } else {
              areaChoice('selectedAreaDrag');
          }
      });
  }
}

// Additional event listeners if needed
// document.addEventListener('DOMContentLoaded', function () {
//   // Your initialization code here
//   initializeToggleSwitches();
//   // other initialization code
// });

// document.addEventListener('DOMContentLoaded', initializeToggleSwitches);

document.addEventListener('DOMContentLoaded', () => {
    const targetElement = document.getElementById('areaSelection');

    const observerOptions = {
        root: null, // Use the viewport as the root
        rootMargin: '0px',
        threshold: 0.1 // Trigger when at least 10% of the element is visible
    };

    function observerCallback(entries, observer) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {

                initializeToggleSwitches();
                observer.unobserve(entry.target); // Stop observing after initialization
            }
        });
    }

    const observer = new IntersectionObserver(observerCallback, observerOptions);
    observer.observe(targetElement);
});

document.addEventListener('click', e => {
  const isDropDownButton = e.target.matches("[data-dropdown-button]");
  const isMainOption = e.target.matches(".main-option");
  const isSubOption = e.target.matches(".sub-option");

  if (isMainOption) {
      e.preventDefault();
      // Update the subtitle
      enableDivs();
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
      mainDropDownRemoves ();
  }

  if (isSubOption) {
    removeCanvasFollow();  
    e.preventDefault();
      // Update the sub subtitle
      const subDropdown = e.target.closest(".subForm .dropdown");
      const label = e.target.getAttribute("data-label");
      subDropdown.querySelector(".sub-subtitle").textContent = label;
      const dataVal = e.target.getAttribute("data-value");
      const value = e.target.getAttribute("data-value");
      
      var edgeDetectionDiv = document.querySelector('.edgeDetection');
      var ocrDiv = document.querySelector('.OCR');
      
      if (edgeDetectionDiv.classList.contains('active')) {
        edgeDetectionChoice(dataVal);
      } else if (ocrDiv.classList.contains('active')) {
        showOCRUpload()
        disableDivs()
      } else {
        showSecondDropChoice(dataVal);;
      }
      
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

function setNewInitialImage(imageData, width, height) {
  initialImage = imageData;
  initialImageWidth = width;
  initialImageHeight = height;

}

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

    // Ensure the canvas is properly sized before drawing the image
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

function getImagePDF(event, callback) {
  var selectedFile = event.target.files[0];
  fileType = selectedFile.type;
  
  if (fileType.includes('pdf')) {
    // Handle PDF file
    pdfformData = new FormData();
    pdfformData.append('file', selectedFile);

    // Call the callback function with FormData containing the PDF file
    if (callback && typeof callback === 'function') {              
      callback(pdfformData, 'pdf');
    }
  } else {
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

      // Create an offscreen canvas
      const offscreenCanvas = document.createElement("canvas");
      const offscreenContext = offscreenCanvas.getContext("2d");

      // Set the canvas dimensions to the new dimensions
      offscreenCanvas.width = newWidth;
      offscreenCanvas.height = newHeight;

      // Draw the image onto the offscreen canvas with the new dimensions
      offscreenContext.drawImage(img, 0, 0, newWidth, newHeight);

      // Get the image data from the offscreen canvas as an ImageData object
      imageDataTempOCR = offscreenContext.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);

      // Call the callback function, if provided, with the ImageData object
      if (callback && typeof callback === 'function') {
        callback(imageDataTempOCR, 'image');
      }
    };
    
    reader.onload = function (e) {
      // Set the source of the image to the data URL
      img.src = e.target.result;
    };
    // Read the selected file as a data URL
    reader.readAsDataURL(selectedFile);
  }
}

// Resets the main image to the original image
function resetInitialImage() {
  
  // Get the source image element and canvas
  currentColorSchemeMain = 'rgbColour';
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

document.addEventListener('DOMContentLoaded', function() {
  
  // Event listener for predImageUploadForm
  const predImageUploadForm = document.getElementById('predImageUploadForm');
  if (predImageUploadForm) {
    
    predImageUploadForm.addEventListener('change', function(event) {
      const fieldset = document.getElementById('predUploadField');
    
      if (fieldset) {
        fieldset.style.borderColor = 'green';
      }
    
      
      getImage(true, event, function() {
        sendPredImage('classImageUpload', clickedimageProcess, 'image/jpeg');
    
      });
    });
  } else {
    console.error('Element with ID predImageUploadForm not found');
  }

  // Event listener for segImageUploadForm
  const segImageUploadForm = document.getElementById('segImageUploadForm');
  if (segImageUploadForm) {
    
    segImageUploadForm.addEventListener('change', function(event) {
      const fieldset = document.getElementById('segUploadField');
    
      if (fieldset) {
        fieldset.style.borderColor = 'green';
      }
    
    
      getImage(true, event, function() {
        sendPredImage('segImageUpload', clickedimageProcess, 'image/jpeg');
      });
    
    });
  } else {
    console.error('Element with ID segImageUploadForm not found');
  }

  // Event listener for ocrImageUploadForm
  const ocrImageUploadForm = document.getElementById('ocrImageUploadForm');
  if (ocrImageUploadForm) {
    ocrImageUploadForm.addEventListener('change', function(event) {
      const fieldset = document.getElementById('ocrUploadField'); // Ensure this is the correct fieldset ID
      if (fieldset) {
        fieldset.style.borderColor = 'green';
      } 
      getImagePDF(event, function() {
        sendPredImage('ocrImageUpload', clickedimageProcess, fileType);
      });
    });
  } else {
    console.error('Element with ID ocrImageUploadForm not found');
  }

});

// Function to get information about the image
function getImageParams() {
  if (initialImage) {
    // Create a new image element
    const image = new Image();
    image.src = initialImage;

    // Create a promise to handle image loading
    return new Promise((resolve) => {
      image.onload = () => {
        // Create an offscreen canvas
        const offscreenCanvas = document.createElement("canvas");
        const offscreenContext = offscreenCanvas.getContext("2d");

        // Set the canvas size to match the initial image size
        offscreenCanvas.width = initialImageWidth;
        offscreenCanvas.height = initialImageHeight;

        // Draw the image onto the offscreen canvas
        offscreenContext.drawImage(image, 0, 0, initialImageWidth, initialImageHeight);

        // Get the image data from the offscreen canvas as an ImageData object
        let imageDataTemp = offscreenContext.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);

        resolve(imageDataTemp);
      };
    });
  } else {
    // Fallback to extracting image data from an element if initial image data URL is not set
    // Get the image element by ID
    const mainImageElement = document.getElementById('sourceImage');

    // Create an offscreen canvas
    const offscreenCanvas = document.createElement("canvas");
    const offscreenContext = offscreenCanvas.getContext("2d");

    // Set the canvas size to match the image size
    offscreenCanvas.width = mainImageElement.naturalWidth;
    offscreenCanvas.height = mainImageElement.naturalHeight;

    // Draw the image onto the offscreen canvas
    offscreenContext.drawImage(mainImageElement, 0, 0);

    // Get the image data from the offscreen canvas as an ImageData object
    let imageDataTemp = offscreenContext.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);

    return Promise.resolve(imageDataTemp);
  }
}

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
        if (!selectedAreaDrag) return; // Allow dragging only if selectedAreaDrag is true.
        isDragging = true;
        startPoint = getMousePos(canvas, event);
    };

    canvas.onmousemove = function(event) {
      if (!selectedAreaDrag || !isDragging) return; // Continue only if selectedAreaDrag is true and the user is dragging
      endPoint = getMousePos(canvas, event);
  
      // Calculate the width and height of the rectangle
      var width = endPoint.x - startPoint.x;
      var height = endPoint.y - startPoint.y;
  
      // Clear the canvas and redraw the image
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(img, 0, 0, canvas.width, canvas.height);
  
      // Draw the selection rectangle with fill and stroke
      context.beginPath();
      context.rect(startPoint.x, startPoint.y, width, height);
      context.fillStyle = "rgba(255, 105, 180, 0.5)"; // Set fill color to semi-transparent
      context.fill(); // Fill the rectangle
      context.strokeStyle = "#ff69b4"; // Set the border color
      context.stroke(); // Draw the rectangle border
  
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
        if (!selectedAreaDrag) return; // Only proceed if selectedAreaDrag is true
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

///// SHOW/REMOVE FUNCTIONS 
function showNoseSeg () {
  const allHoverSquare = document.querySelector('#segUploadField');
  allHoverSquare.style.display = 'flex';  
}

function showOCRUpload () {
  led.classList.remove('ready')
  led.classList.add('broken')
  const allHoverSquare = document.querySelector('#ocrUploadField');
  allHoverSquare.style.display = 'flex';  
}

function showTessText() {
  const displayedTextContainer = document.getElementById("ocr");
  displayedTextContainer.style.display = 'flex'; // Ensure the container is visible
}

function showOCRImage() {
  const imageContainer = document.getElementById("image-container");
  imageContainer.style.display = 'block'; // Ensure the container is visible
}

function displayPageCount(currentPage, totalPages) {
  const pageCountElement = document.getElementById('pageCount');
  pageCountElement.textContent = `Showing page ${currentPage} of ${totalPages}`;
}

// Shows the Area Selected Div
function showAreaChoice () {
  //Hide area Elements
  initializeToggleSwitches();
  const selectedAreaType = document.getElementById("areaSelection"); 
  selectedAreaType.style.display = "flex";
  selectedAreaType.style.flexDirection = "column";
  selectedAreaType.style.justifyContent = "centre";
  selectedAreaType.style.margin = "5px";
}

// Hides the Areas that display what is in the hover box and their resutls
function showSlider(newLabelText,isfftSlider) {
  const allSliderElem = document.getElementById('numberSlider');
  const sliderInput = allSliderElem.querySelector('.segSlider'); // Get the existing slider input
  const currentSliderValue = allSliderElem.querySelector('#sliderValue').textContent; // Get the current slider value
  
  // Update the innerHTML to include both the new label text and preserve the slider input
  allSliderElem.innerHTML = `${newLabelText}: <span id="sliderValue">${currentSliderValue}</span>`;
  allSliderElem.insertBefore(sliderInput, allSliderElem.firstChild); // Re-insert the slider input at the beginning

  if (isfftSlider) {
    fourrMaxCutoff = maxCutoffFrequency(initialImageHeight, initialImageWidth)
    sliderInput.max = fourrMaxCutoff
  } else {
    sliderInput.max = 10;
  }

  // Re-apply styles and make it visible
  allSliderElem.style.display = 'flex';
  led.classList.add('ready');

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
  allCanvasArea.style.border ='none';
}

function showHoverSquare () {
  const allHoverSquare = document.querySelector('.hoverSquare');
  allHoverSquare.style.display = 'flex'; 
}

function removeHoverSquare() {
  const allHoverSquares = document.querySelector('.hoverSquare');
  allHoverSquares.style.display = 'none';
  };

function showHoverBox () {
  const hoverBoxselector = document.querySelector("#showHoverBox");
  hoverBoxselector.style.display = 'flex';
}

function removeHoverBox() {
  const hoverBoxselector = document.querySelector("#showHoverBox");
  const canvasElements = hoverBoxselector.querySelectorAll('canvas, div');

  // Check if any of the canvas or div elements are visible
  let visibleCanvasOrDiv = false;
  canvasElements.forEach(element => {
      if (element.style.display !== 'none') {
          visibleCanvasOrDiv = true;
      }
  });

  // Only hide the hover box if no visible canvas or div elements are present
  if (!visibleCanvasOrDiv) {
      hoverBoxselector.style.display = 'none';
      hoverBoxselector.style.outline = 'none'; // Remove any outline
      hoverBoxselector.style.border = 'none'; // Remove any border
      hoverBoxselector.style.boxShadow = 'none'; // Remove any box shadow

      // Remove outlines from child elements
      const childElements = hoverBoxselector.querySelectorAll('*');
      childElements.forEach(child => {
          child.style.outline = 'none';
          child.style.border = 'none';
          child.style.boxShadow = 'none';
      });

  }
}

function showHoverSize () {
  // Shows size of Hover Square
  const hoverSizeSelector = document.querySelector("#hoverSizeSelector");
  hoverSizeSelector.style.display = 'flex';

  boxWidth = 100;
  boxHeight = 100;
  document.getElementById('boxWidth').value = '';
  document.getElementById('boxHeight').value = '';
  document.getElementById('boxWidth').placeholder = 'Box Width (Default 100)';
  document.getElementById('boxHeight').placeholder = "Box Height (Default 100)";

  updateHoverBox()
}

function showImageResize() {
  const imageResizeSelector = document.querySelector("#imageResize");
  imageResizeSelector.style.display = 'flex';
  imageResizeSelector.style.flexDirection = "column";
  selectedImageWidth = '';
  selectedImageHeight = '';
  document.getElementById('imageSelectedBoxWidth').value = '';
  document.getElementById('imageSelectedBoxHeight').value = '';
  updateImageSize ();
}

function showRotateAngle () {
  const selectedRotateElement = document.querySelector("#rotateAngle");
  selectedRotateElement.style.display = 'flex';
  selectedRotateAngle = 45;
  document.getElementById('rotateAngleSelection').value = '';
  document.getElementById('rotateAngleSelection').placeholder = 'Rotate Angle (Default 45)';
  rotateAngle ();
}

function showCameraButton() {
  const selectedCameraButton = document.querySelector("#aslButton");
  if (selectedCameraButton) {
    selectedCameraButton.style.display = 'block'; // Show the fieldset
    console.log('ASL Button shown');
  } else {
    console.log('ASL Button not found');
  }
}

function showSwapColour () {
  const selectedRotateElement = document.querySelector("#colourSelection");
  selectedRotateElement.style.display = 'flex';
  colourChoice();
}

function showSimpleThresh () {
  const selectedSimpleThreshElement = document.querySelector("#simpleThresh");
  selectedSimpleThreshElement.style.display = 'flex';
}

function showTranslate () {
  const selectedTranslateElement = document.querySelector("#translateDistanceSelector");
  selectedTranslateElement.style.display = 'flex';
  document.getElementById('translateX').placeholder = 'X Direction (Default 50)';
  document.getElementById('translateY').placeholder = "Y Direction (Default 30)";
  document.getElementById('translateX').value = '';
  document.getElementById('translateY').value = '';
  translateXDist = 50;
  translateYDist = 30;
  translateDistances = [50, 30];

  updateTranslateDist();
}

function showThreshVals () {
  const selectedThreshElement = document.querySelector("#threshVals");
  selectedThreshElement.style.display = 'flex';
  document.getElementById('thresholdValue').placeholder = "Threshold Value (Default 127)";
  document.getElementById('thresholdMax').placeholder = "Threshold Max (Default 255)";
  document.getElementById('thresholdValue').value = '';
  document.getElementById('thresholdMax').value = '';
  newThresholdValue = 127;
  newThresholdMax = 255;

  updateThresholdVals();
}

function showAffineTransform () {
  const selectedAffineTranformElement = document.querySelector("#affineTransfromSelector");
  selectedAffineTranformElement.style.display = 'flex';

  document.getElementById('affineAngle').placeholder = "Rotation Angle (Default 45)";
  document.getElementById('affineScale').placeholder = "Scaling Factor (Default 1.5)";
  document.getElementById('affineAngle').value = '';
  document.getElementById('affineScale').value = '';
  affineAngle = 45;
  affineScale = 1.5;
  affineTransformChoices = [affineAngle,affineScale]
  updateAffineTransform();
}

function showSmoothingKernel () {
  const selectedThreshElement = document.querySelector("#smoothingKernel");
  selectedThreshElement.style.display = 'flex';
}

function showSharpKernel () {
  const selectedThreshElement = document.querySelector("#sharpeningKernel");
  selectedThreshElement.style.display = 'flex';
}

function showLoading() {
  const selectedLoadedElement = document.querySelector("#loader-container");
  selectedLoadedElement.style.display = 'flex';
}

function removeLoading() {
  const selectedLoadedElement = document.querySelector("#loader-container");
  selectedLoadedElement.style.display = 'none';
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
  contourBoundingBoxChoice();
}

function showcontourFeatureChoice () {
  const selectedFeaturehElement = document.querySelector("#contourFeatureSelection");
  selectedFeaturehElement.style.display = 'flex';
  contourFeatureSelection = "contourArea";
  contourFeatureChoice()
  
}

function showCanvas () {
  const allCanvasArea = document.querySelector('.allCanvas');
  allCanvasArea.style.display = 'grid';
  
}

function showCanvasFollow() {
  const selectedCanvasElement = document.querySelector('#myCanvasFollow');
  selectedCanvasElement.style.display = 'block';
  selectedCanvasElement.style.border = '2px solid black'; // Ensure border is set
}

function showClassPredText () {
  // Select the paragraph element by its ID
  var paragraph = document.getElementById('classPreds');

  // Update the text content of the paragraph
  paragraph.style.display = 'flex';
}

function removeCLassPredText () {
// Select the paragraph element by its ID
var paragraph = document.getElementById('classPreds');

// Update the text content of the paragraph
paragraph.style.display = 'none';
}

function showSegPreds (SegPreds) {
predDiv = document.getElementById("classPreds")
predDiv.innerHTML = "";


var newParagraph = document.createElement("p");
newParagraph.textContent = SegPreds;
predDiv.appendChild(newParagraph);
showClassPredText ()
}

function showClassPreds(binPreds,multiPreds) {
swapClassPredsToText (binPreds,multiPreds)
showClassPredText ()
}

function showAdaptThreshVals() {
  const selectedThreshElement = document.querySelector("#adaptiveThresh");
  selectedThreshElement.style.display = 'flex';

  // Reset values and update placeholders to ensure they follow a uniform pattern
  document.getElementById('maxValue').placeholder = "Threshold Max (Max 255)";
  document.getElementById('maxValue').value = '';
  adaptiveMaxValue = 255;

  document.getElementById('blockSize').placeholder = "Block Size (e.g., 3, 5, 7)";
  document.getElementById('blockSize').value = '';
  adaptiveBlockSize = 3;

  document.getElementById('constantC').placeholder = "Constant C (e.g., -5, 0, 5)";
  document.getElementById('constantC').value = '';
  adaptiveConstant = 0;

  document.getElementById('adaptiveMethod').value = 'meanAdapt';  // Assuming 'meanAdapt' corresponds to 'Mean'
  adaptiveMethod = 'meanAdapt';  // This setting should correspond to the value attribute in the select option

  document.getElementById('thresholdType').value = 'binaryAdapt';  // Assuming 'binaryAdapt' corresponds to 'Binary'
  adaptiveThresholdType = 'binaryAdapt';  // This setting should correspond to the value attribute in the select option

  // Call the update function to apply changes and handle form validation or further processing
  adaptiveThreshChoice();
}

function showNextFreeCanvas(squareType, nextFreeRow) {

  showHoverBox ();
  const subCanvasi = document.getElementById(`subCanvas${nextFreeRow}`);
  const myCanvasDivi = document.getElementById(`myCanvasDiv${nextFreeRow}`);
  const mainImageCanvasi = document.getElementById(`mainImageCanvas${nextFreeRow}`);
  const selectedResetCanvasButton = document.querySelector(`#canvasButtons${nextFreeRow}`);
  let returnSquare;
  selectedResetCanvasButton.style.display = 'flex'

  if (squareType == 'mainCanvas') {
    mainImageCanvasi.style.display = 'block'
    // selectedResetCanvasButton.style.display = 'block'
    returnSquare =  mainImageCanvasi.id 
  } else if (squareType == 'subCanvas') {
    subCanvasi.style.display = 'block'
    returnSquare =  subCanvasi; 
  } else if (squareType == 'divCanvas') {
    myCanvasDivi.style.display = 'block'
    returnSquare =  myCanvasDivi;  
  }
  return returnSquare  
}

function showSecondDropChoice(subChoice) {
  removeCamera();

  secondDropDownChoice = subChoice
  secondaryDropDownRemoves();
  let entireList = ["resize","translate","FftSpectrum","FftFilter","clusterSeg","binaryClass","multiClass","threshSeg","semantic","customSemantic","ASL"];
  let selectedList = ["crop"];
  let choiceList = ["grayscale","rotate","swapColour","swapColour","simpleThresh","adaptThresh","otsuThresh","imageHist","histEqua","affine",
  "identityKernel","smoothingKernel","sharpeningKernel","edgeDetectionKernel","morphologicalKernel","frequencyDomainKernel","customKernel",
  "drawContours",'drawShapes',"contourFeatures","boundingFeatures","edgeDetection","segmentationOpsDropDown","identifyShapes"]; 
            
  if (choiceList.includes(secondDropDownChoice)) {         
    selectedAreaStamp = false;
    selectedAreaStamp = false;
    selectedAreaDrag = false;
    showAreaChoice();
    showCanvas ();
    // Check if Selected Area Type is Ticked 
    if (selectedAreaStamp) {
      showHoverBox ();
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
       
      } else if (secondDropDownChoice == 'smoothingKernel') {
        showSmoothingKernel ()
      } else if (secondDropDownChoice == 'sharpeningKernel') {
        showSharpKernel ()      
      } else if (secondDropDownChoice == 'edgeDetectionKernel') {
        showEdgeKernel ()
      } else if (secondDropDownChoice == 'morphologicalKernel') {
        showMorphologicalKernel ()
      } else if (secondDropDownChoice == 'customKernel') {
        showCustomKernel ()
      } else if (secondDropDownChoice == 'drawShapes') {
        showShapeChoice()
      } else if (secondDropDownChoice == 'contourFeatures') {
        showcontourFeatureChoice()
      } else if (secondDropDownChoice == 'boundingFeatures') {
        showBoundingBoxChoice()
      } else if (secondDropDownChoice == 'identifyShapes') {

      }              

  } else if (selectedList.includes(secondDropDownChoice))  {
      showAreaChoice();
      showCanvas ();
      placeImageCanvas = true;
          

    
    } else if (entireList.includes(secondDropDownChoice))  {
      
      selectedAreaStamp = false;
      selectedAreaDrag = false;
      entireImage = true;
      if (secondDropDownChoice == 'translate') {
        showTranslate ()
        placeImageCanvas = false;
      } else if (secondDropDownChoice == "resize") {
        showImageResize()
        placeImageCanvas = false;
      } else if (secondDropDownChoice == "FftSpectrum") {
        placeImageCanvas = true
        checkIfReadyToClick();
      } else if (secondDropDownChoice == "FftFilter") {
        showFftFilter ()
        placeImageCanvas = true
      } else if (secondDropDownChoice == 'clusterSeg') {
        showClusterSeg();
      } else if (secondDropDownChoice == 'watershed') {
        // do nothing
      } else if (secondDropDownChoice == 'binaryClass') {
        showBinClass()
        led.classList.add('broken');
      } else if (secondDropDownChoice == 'multiClass') {
        showMultiClass()
        led.classList.add('broken');
      } else if (secondDropDownChoice == 'threshSeg') {
        showSlider('Number of Thresholds')
      } else if (secondDropDownChoice == 'semantic') {
        led.classList.add('ready');
        checkIfReadyToClick();
      } else if (secondDropDownChoice == 'customSemantic') {
        showNoseSeg();
        led.classList.add('broken');
      } else if (secondDropDownChoice == 'ASL') {
        console.log('ASL BUTTON SHOW')
        showCameraButton();
      }         
           
      
    }
} 

///// REMOVE FUNCTIONS

function removeCanvasFollow () {
  const selectedCanvasElement = document.querySelector('#myCanvasFollow');
  selectedCanvasElement.style.display = 'none';
}

function removeMainCanvas1 () {
  const selectedCanvasElement = document.querySelector('#mainImageCanvas1');
  selectedCanvasElement.style.display = 'none';
}

function removeMainCanvas2 () {
  const selectedCanvasElement = document.querySelector('#mainImageCanvas2');
  selectedCanvasElement.style.display = 'none';
}

function removeMainCanvas3 () {
  const selectedCanvasElement = document.querySelector('#mainImageCanvas3');
  selectedCanvasElement.style.display = 'none';
}

function removeSubCanvas1 () {
  const selectedCanvasElement = document.querySelector('#subCanvas1');
  selectedCanvasElement.style.display = 'none';
}

function removeSubCanvas2 () {
  const selectedCanvasElement = document.querySelector('#subCanvas2');
  selectedCanvasElement.style.display = 'none';
}

function removeSubCanvas3 () {
  const selectedCanvasElement = document.querySelector('#subCanvas3');
  selectedCanvasElement.style.display = 'none';
}

function removeCanvasDiv1 () {
  const selectedCanvasElement = document.querySelector('#myCanvasDiv1');
  selectedCanvasElement.style.display = 'none';
}

function removeCanvasDiv2 () {
  const selectedCanvasElement = document.querySelector('#myCanvasDiv2');
  selectedCanvasElement.style.display = 'none';
}

function removeCanvasDiv3 () {
  const selectedCanvasElement = document.querySelector('#myCanvasDiv3');
  selectedCanvasElement.style.display = 'none';
}

function removeResetCanvasButton1() {
  const selectedCanvasElement = document.querySelector('#canvasButtons1');
  selectedCanvasElement.style.display = 'none';

  // Check if there are any visible child divs inside the showHoverBox
  const hoverBox = document.querySelector("#showHoverBox");
  const childDivs = hoverBox.querySelectorAll("div.canvasButtons");

  let visibleChildDivs = false;
  childDivs.forEach(div => {
      if (div.offsetWidth > 0 && div.offsetHeight > 0) {
          visibleChildDivs = true;
      }
  });

  // If no visible child divs are present, hide the showHoverBox
  if (!visibleChildDivs) {
      hoverBox.style.display = 'none';
  }
}

function removeResetCanvasButton2() {
  const selectedCanvasElement = document.querySelector('#canvasButtons2');
  selectedCanvasElement.style.display = 'none';

  // Check if there are any visible child divs inside the showHoverBox
  const hoverBox = document.querySelector("#showHoverBox");
  const childDivs = hoverBox.querySelectorAll("div.canvasButtons");

  let visibleChildDivs = false;
  childDivs.forEach(div => {
      if (div.offsetWidth > 0 && div.offsetHeight > 0) {
          visibleChildDivs = true;
      }
  });

  // If no visible child divs are present, hide the showHoverBox
  if (!visibleChildDivs) {
      hoverBox.style.display = 'none';
  }
}

function removeResetCanvasButton3() {
  const selectedCanvasElement = document.querySelector('#canvasButtons3');
  selectedCanvasElement.style.display = 'none';

  // Check if there are any visible child divs inside the showHoverBox
  const hoverBox = document.querySelector("#showHoverBox");
  const childDivs = hoverBox.querySelectorAll("div.canvasButtons");

  let visibleChildDivs = false;
  childDivs.forEach(div => {
      if (div.offsetWidth > 0 && div.offsetHeight > 0) {
          visibleChildDivs = true;
      }
  });

  // If no visible child divs are present, hide the showHoverBox
  if (!visibleChildDivs) {
      hoverBox.style.display = 'none';
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

function removeSubCanvasElements() {
  // Select all elements with the class 'subCanvas'
  const elements = document.querySelectorAll('.subCanvas');
  
  // Loop through the NodeList and remove each element
  elements.forEach(element => {
    element.style.display = 'none';
  });
}

function removeAllCanvas () {
  removeHoverBox ();
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
  removeSubCanvasElements();
}
///// COLLECTED REMOVES

function uploadRemoves () {
  resetMainDrop();
  resetAllSecondDrops();
  hideSecondForms();
  hideSecondFormsParams();
  removeAllCanvas();
  removeCamera();

  selectedAreaStamp = false;
}

function mainDropDownRemoves () {
  resetMainDrop();
  resetAllSecondDrops();
  hideSecondForms();
  hideSecondFormsParams();
  removeCamera();
  selectedAreaStamp = false;
}

function secondaryDropDownRemoves () {
  hideSecondFormsParams();
  selectedAreaStamp = false;
}

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

///// CHOICE FUNCTIONS
// function that does the choice of the area radio buttons
function areaChoice(areaMethodSelection) {

  if (areaMethodSelection == 'selectedAreaStamp') {
      // The "Selected Area" radio button is selected
      selectedAreaStamp = true;
      selectedAreaDrag = false;
      entireImage = false;
      placeImageCanvas = true;
      showHoverBox ();
      showCanvasFollow();
      showHoverSize();
      showHoverSquare();
    } else if (areaMethodSelection == 'selectedAreaDrag')  {
      removeHoverSquare();
      removeCanvasFollow();
      removeHoverBox();
      selectedAreaStamp = false;
      selectedAreaDrag = true;
      entireImage = false;
      placeImageCanvas = true;
    } else {
      // Another radio button is selected
      removeHoverSquare();
      removeCanvasFollow();
      removeHoverBox();
      entireImage = true;
      selectedAreaStamp = false;
      selectedAreaDrag = false;
      placeImageCanvas = false;
    }
  }

function updateTranslateDist(isbuttonClicked = false) {
  const fieldset = document.getElementById('translateDistanceSelector');
  if (isbuttonClicked) {  
    translateXDist = document.getElementById("translateX").value;
    translateYDist = document.getElementById("translateY").value;
  }
  // Check if both inputs are positive integers
  if (isInteger(translateXDist) && isInteger(translateYDist)) {
    // Change the border color to green
    fieldset.style.borderColor = 'green';
    translateDistances = [translateXDist,translateYDist]
    
  } else {
      // Change the border color to red and show an alert
      fieldset.style.borderColor = 'red';
      if (isbuttonClicked) {
        alert('Please enter two integers for X and Y.');
      }
  }   
}

function updateAffineTransform (isbuttonClicked=false) {
  const fieldset = document.getElementById('affineTransfromSelector');
  if (isbuttonClicked) {
    affineAngle = document.getElementById("affineAngle").value;
    affineScale = document.getElementById("affineScale").value;
    }

  // Check if both inputs are positive integers
  if (isNumber(affineAngle) && (isFloat(affineScale) || isInteger(affineScale))) {
    // Change the border color to green
    fieldset.style.borderColor = 'green';
    affineTransformChoices = [affineAngle,affineScale]
  } else {
      // Change the border color to red and show an alert
      fieldset.style.borderColor = 'red';
      if (isbuttonClicked) {
        alert('Please enter two integers for X and Y.');
      }
  }  

  
}

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

function edgeDetectionChoice(edgeChoicce) {
  // Get the selected radio button value
  selectedEdgeDetection = edgeChoicce;
  secondDropDownChoice = 'edgeDetection';
  showAreaChoice();
  showCanvas ();
  
  // Check if Selected Area Type is Ticked 
  if (selectedAreaStamp) {
    showHoverBox ();
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

  const fieldsetPred = document.getElementById('predUploadField');
  fieldsetPred.style.borderColor = 'red';
}

function multiClass() {
  clickedimageProcess = 'multiClass'
  showClassButton();
  clickedClassModel = document.querySelector('input[name="multiClassSelection"]:checked').value;
  const fieldset = document.getElementById('multiModelSelection');
  fieldset.style.borderColor = 'green';

  const fieldsetPred = document.getElementById('predUploadField');
  fieldsetPred.style.borderColor = 'red';

}

function objectDetectionChoice() {
  clickedimageProcess = 'objectDetection'
  showClassButton();
  objDetModel = document.querySelector('input[name="objectDetectionModel"]:checked').value;
  const fieldsetPred = document.getElementById('predUploadField');
  fieldsetPred.style.borderColor = 'red';
  
}

function clusterSegChoice() {
  const fieldset = document.getElementById('clusterSeg');
  clusterSeg = document.querySelector('input[name="clusterSegSelection"]:checked').value;
  isfftSlider = false;
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
  entireImage = true;
  fieldset.style.borderColor = 'green';
  isfftSlider = true;
  showSlider('Choose Filter Threshold',isfftSlider)
  
}

function simpleThreshChoice() {
  const fieldset = document.getElementById('simpleThresh');
  // Get all radio buttons with the name "thresholdMethod"
  selectedSimpleThresholdMethod = document.querySelector('input[name="thresholdMethod"]:checked').value;
  fieldset.style.borderColor = 'green'; 
}

function updateThresholdVals(isbuttonClicked=false) {
  const fieldset = document.getElementById('threshVals')
  // Get the values from the input fields
  if (isbuttonClicked) {
    newThresholdValue = document.getElementById("thresholdValue").value;
    newThresholdMax = document.getElementById("thresholdMax").value;
    }

  
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
    if (isbuttonClicked) {
      alert('Please enter two positive integers for Threshold Value and Max.');
    }
  }  
}
  
function adaptiveThreshChoice(isButtonClicked) {
  let maxValueInput = '';
  let methodInput = '';
  let thresholdTypeInput = '';
  let blockSizeInput = '';
  let constantCInput = '';
  
  if (isButtonClicked) {
      // Only assign new values from the form if the button has been clicked
      maxValueInput = document.getElementById('maxValue').value;
      methodInput = document.getElementById('adaptiveMethod').value;
      thresholdTypeInput = document.getElementById('thresholdType').value;
      blockSizeInput = document.getElementById('blockSize').value;
      constantCInput = document.getElementById('constantC').value;
  }
  
  // Retrieve each form value or use default if the input is empty
  adaptiveMaxValue = maxValueInput ? maxValueInput : adaptiveMaxValue;
  adaptiveMethod = methodInput ? methodInput : adaptiveMethod;
  adaptiveThresholdType = thresholdTypeInput ? thresholdTypeInput : adaptiveThresholdType;
  adaptiveBlockSize = blockSizeInput ? blockSizeInput : adaptiveBlockSize;
  adaptiveConstant = constantCInput ? constantCInput : adaptiveConstant;

  // Ensure numeric values are correctly converted from strings
  adaptiveMaxValue = maxValueInput ? parseInt(maxValueInput) : adaptiveMaxValue;
  adaptiveBlockSize = blockSizeInput ? parseInt(blockSizeInput) : adaptiveBlockSize;
  adaptiveConstant = constantCInput ? parseInt(constantCInput) : adaptiveConstant;

  
  // Validate the provided or default values
  if (isPositiveInteger(adaptiveMaxValue) && isInteger(adaptiveConstant) && isOddInteger(adaptiveBlockSize) && adaptiveBlockSize > 1) {
      // Update global parameters only if validation is successful
      adaptiveParamaters = [adaptiveMaxValue, adaptiveMethod, adaptiveThresholdType, adaptiveBlockSize, adaptiveConstant];
      document.getElementById('adaptiveThresh').style.borderColor = 'green';
  } else {
      // Set border color to red to indicate error and show alert if the button was clicked
      document.getElementById('adaptiveThresh').style.borderColor = 'red';
      if (isButtonClicked) {
          alert('Please ensure Max Value is positive integers, Constant C is an integer and Block Size is Odd and >1');
      }
  }
}

function smoothingKernelChoice() {
  const fieldset = document.getElementById('smoothingKernel');
  // Get the selected radio button value
  selectedKernel = document.querySelector('input[name="smoothingKernelSelection"]:checked').value;
  fieldset.style.borderColor = 'green';
}

function sharpKernelChoice() {
  const fieldset = document.getElementById('sharpeningKernel');
  // Get the selected radio button value
  selectedKernel = document.querySelector('input[name="sharpKernelSelection"]:checked').value;
  fieldset.style.borderColor = 'green';
}

function edgeKernelChoice() {
  const fieldset = document.getElementById('edgeKernel');
  // Get the selected radio button value
  selectedKernel = document.querySelector('input[name="edgeKernelSelection"]:checked').value;
  fieldset.style.borderColor = 'green';
}

// Updates the size of the hover box
function updateHoverBox(isbuttonClicked=false) {
  const hoverElement = document.querySelector(".hoverSquare");
  const fieldset = document.getElementById('hoverSizeSelector');
  const sourceImage = document.getElementById("sourceImage");
  // Assuming you want to update the main element with the "hoverSquare" class
  if (isbuttonClicked) {
    boxWidth = document.getElementById("boxWidth").value;
    boxHeight = document.getElementById("boxHeight").value;
  }

  const imageWidth = sourceImage.naturalWidth;
  const imageHeight = sourceImage.naturalHeight;

  // Check if both inputs are positive integers
  if (isPositiveInteger(boxWidth) && isPositiveInteger(boxHeight) && (boxWidth < imageWidth) && (boxHeight <imageHeight )) {
    // Change the border color to green
    fieldset.style.borderColor = 'green';
    // Show the hover box if selectedArea is true
    if (selectedAreaStamp || selectedAreaDrag) {
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
        if (isbuttonClicked) {
          alert(`Please enter two positive integers for width < ${imageWidth} and height < ${imageHeight} `);
        };
    } 
}

///// CANVAS FUNCTIONS
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

function getNextFreeRow() {
  for (let i = 1; i <= 3; i++) {
      const mainImageCanvas = document.getElementById(`mainImageCanvas${i}`);
      const subCanvas = document.getElementById(`subCanvas${i}`);
      const divCanvas = document.getElementById(`myCanvasDiv${i}`);

      const mainImageCanvasStyle = window.getComputedStyle(mainImageCanvas).display;
      const subCanvasStyle = window.getComputedStyle(subCanvas).display;
      const divCanvasStyle = window.getComputedStyle(divCanvas).display;

      if (mainImageCanvasStyle === 'none' && subCanvasStyle === 'none' && divCanvasStyle === 'none') {
          return i; // Return row index (1-based)
      }
  }

  // If all rows are occupied, return null
  return null;
}

function disableCanvasInteraction(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (canvas) {
      canvas.style.pointerEvents = 'none';
      canvas.style.backgroundColor = '#d3d3d3';
      canvas.style.opacity = '0.5';
  } else {
      console.error('Canvas element not found');
  }
}

function enableCanvasInteraction(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (canvas) {
      canvas.style.pointerEvents = 'auto';
      canvas.style.backgroundColor = ''; // Reset to default background color
      canvas.style.opacity = '1'; // Reset to full opacity
  } else {
      console.error('Canvas element not found');
  }
}

///// TECHINCAL RESULT DISPLAY FUNCTIONS

function createPlotlyHistogram (histdata,nextFreeRow) {
  
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

  

  // let nextFreeCanvasId = getNextFreeCanvas('divCanvas');
  let nextFreeCanvasId = showNextFreeCanvas('divCanvas', nextFreeRow);
  Plotly.newPlot(nextFreeCanvasId, histDataPlotly, layout);
}

function createFftThresh (data,nextFreeRow) {
  // let nextFreeCanvasId = getNextFreeCanvas('divCanvas');
  let nextFreeCanvasId = showNextFreeCanvas('subCanvas', nextFreeRow);
  drawImageInCanvas(data.fftThresh, nextFreeCanvasId.id)
}

function setColumnsForRow(nextFreeRow, numColumPerRow) {
  const rowId = `row${nextFreeRow}`;
  const gridRowNum = document.getElementById(rowId);

  gridRowNum.style.gridTemplateColumns = `repeat(${numColumPerRow}, 1fr)`;

  // Print the current value of gridTemplateColumns
  const currentGridTemplateColumns = gridRowNum.style.gridTemplateColumns;
}

function updateImageSize (isbuttonClicked=false) {
  const fieldset = document.getElementById('imageResize');
  selectedImageWidth = document.getElementById("imageSelectedBoxWidth").value;
  selectedImageHeight = document.getElementById("imageSelectedBoxHeight").value;
  
  const height = selectedImageHeight;
  const width = selectedImageWidth;

  // Check if both inputs are not empty and are numbers
    // Check if both inputs are positive integers
    if (selectedImageWidth !== '' && selectedImageHeight !== '' && isPositiveInteger(width) && isPositiveInteger(height)) {
      // Change the border color to green
      fieldset.style.borderColor = 'green';
      secondformParamActiveList.push(selectedImageWidth,selectedImageHeight)
  } else {
      // Change the border color to red and show an alert
      fieldset.style.borderColor = 'red';
      if (isbuttonClicked) {
        alert('Please enter two positive integers for width and height.')
      };
  }  
}

function rotateAngle(isbuttonClicked=false) {
  const fieldset = document.getElementById('rotateAngle');

  if (isbuttonClicked) {
    selectedRotateAngle = document.getElementById("rotateAngleSelection").value;
  }
  
  if (selectedRotateAngle != '' && isNumber(selectedRotateAngle)) {
      // Change the border color to green
      fieldset.style.borderColor = 'green';
  } else {
      // Change the border color to red and show an alert
      fieldset.style.borderColor = 'red';
      if (isbuttonClicked) {
        alert('Please enter a valid float for the Rotation Angle.');
      }
  }  
}

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
  // Change the color of the <p> element to green
  const paragraph = document.getElementById('customKernelText');
  const confirmationMessage = document.getElementById('confirmationMessage');
  
  paragraph.style.color = 'green';
  selectedKernel = data;
  checkIfReadyToClick()
  
  confirmationMessage.style.display = 'block';

  setTimeout(() => {
    confirmationMessage.style.display = 'none';
  }, 3000);
  
}

///// MAIN IMAGE TO CANVAS HANDLING FUNCTIONS 

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
  imageDataWidth = selectedImageInfo[1];
  imageDataHeight = selectedImageInfo[2];

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
  imageData = ctx.getImageData(0, 0, canvasTrial.width, canvasTrial.height);
  return [imageData, canvasTrial.width, canvasTrial.height];
}

// Event to display square when in image container
document.querySelector('#imageCanvas').addEventListener('mouseenter', function () {

  if (selectedAreaStamp) {
      document.querySelector('.hoverSquare').style.display = 'flex';
      }
  });

// Event to remove square when out image container
document.querySelector('#imageCanvas').addEventListener('mouseleave', function () {
  document.querySelector('.hoverSquare').style.display = 'none';

});


function jsonReplaceMainImg(data) {
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

function swapClassPredsToText (binPreds,multiPreds) {
  let classText = '';
  predDiv = document.getElementById("classPreds")
  predDiv.innerHTML = "";

  // Check if binPreds is not False (i.e., it is a valid number)
  if (binPreds !== false) {
    if (binPreds > 0.5) {
      // If binPreds is greater than 0.5, classify as dog
      classText = `The Image is of a dog with ${binPreds.toFixed(2)} probability`;
    } else {
      // Otherwise, calculate the probability for cat and classify as cat
      let catProb = 1 - binPreds;
      classText = `The Image is of a cat with ${catProb.toFixed(2)} probability`;
    }
    var newParagraph = document.createElement("p");
    newParagraph.textContent = classText;
    predDiv.appendChild(newParagraph);
  } else {
    
    
    // If binPreds is False, process multiple predictions
    for (let i = 0; i < multiPreds.length; i++) {
      let classPredI = multiPreds[i][1]; // Assuming the class name is at index 1
      let classProbI = multiPreds[i][2].toFixed(2);; // Assuming the probability is at index 2

      // Using the rounded value in the template string
      let tempText = ` - The class probabilities are ${classPredI} with ${classProbI} probability`;

      var newParagraph = document.createElement("p");
      newParagraph.textContent = tempText;
      predDiv.appendChild(newParagraph);
    }
  }
  
}

///// IMAGE SEGMENTATION FUNCTIONS
function initializeDataTable() {
  // If the DataTable already exists, destroy it to reinitialize later
  if ($.fn.DataTable.isDataTable('#dataTable')) {
      $('#dataTable').DataTable().destroy();
  }

  // Assuming showSegTable() prepares the data or layout for DataTable
  showSegTable();

  // Explicitly set the 'All' checkbox to checked
  document.getElementById('segAllCheck').checked = true;

  // Set the state of all checkboxes in the 'segOptionCheck' div to match the 'All' checkbox
  document.querySelectorAll('#segOptionCheck input[type="checkbox"]').forEach(function(checkbox) {
      checkbox.checked = true;  // Directly set all to checked
  });

  // Initialize the DataTable
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
      pageLength: 5,
      lengthMenu: [ [5, 10, 25, 50, -1], [5, 10, 25, 50, "All"] ],
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
    // Retrieve the value from the search bar and convert it to a number
    var probValue = parseFloat($('#minProbability').val());

    // Add a custom search function to DataTables
    $.fn.dataTable.ext.search.push(function(settings, data, dataIndex) {
        // Access the value from the specific column (index 2, assuming it's the third column) and convert it to a number
        var value = parseFloat(data[2]);

        // Check if the value is greater than the probability value entered in the search bar
        return value > probValue;
    });

    // Redraw the DataTable to apply the search filter
    table.draw();

    // Remove the custom search function after filtering to avoid affecting future filtering
    $.fn.dataTable.ext.search.pop();
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



$('#processBtn').click(function() {
  var selectedRowsData = table.rows({ selected: true, search: 'applied' }).data();
  var selectedRowNums = selectedRowsData.map(function(data) {
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
        // Check if main image data is provided
        if (response.img) {
            // Update the main image display
            jsonReplaceMainImg(response);
        }

        // Check if there is a URL provided for downloading the zip file
        if (response.zip_url) {
            // Automatically trigger the download of the zip file
            window.location.href = response.zip_url;
        }
      },
      error: function(err) {
          console.error('Error processing: ' + JSON.stringify(err));
      }
  });
});
}

///// SEND AJAX FUNCTIONS
// Initiates sending the image to python
document.querySelector('#imageCanvas').addEventListener('click', async function (e) {
  const ledElement = document.getElementById('led');
  const computedStyle = window.getComputedStyle(ledElement);
  const ledbackgroundColor = computedStyle.backgroundColor;
  
  if (ledbackgroundColor === 'rgb(255, 0, 0)') {
      alert("Please Ensure All Selections Are Green To Continue");
      return;
  }

  if (selectedAreaStamp) {
      imageData = imageData.data;
      sendImageSnippet(imageData, imageDataHeight, imageDataWidth, secondDropDownChoice);
  }

  if (selectedAreaDrag) {
      return;
  }

  if (entireImage) {
      try {
          let entireImageData = await getImageParams();
          imageData = entireImageData.data;
          imageDataHeight = entireImageData.height;
          imageDataWidth = entireImageData.width;

          // Send the image snippet
          sendImageSnippet(imageData, imageDataHeight, imageDataWidth, secondDropDownChoice);
      } catch (error) {
          console.error("Error getting image parameters:", error);
      }
  } else {
      // Handle cases where `entireImage` is not true
      // Get a snippet sized image
      // Ensure you have logic here if necessary
  }
});

function sendImageSnippet(imageData, imageHeight, imageWidth, selectedImageProcess) {
  led.classList.add('thinking');
  const imgDisplayColumn = document.getElementById('mainImage');
  imgDisplayColumn.classList.add('imgDisplayColumn'); // Add the class dynamically
  imgDisplayColumn.style.position = 'relative'; // Apply position relative
  imgDisplayColumn.style.display = 'inline-block'; // Apply display inline-block
  disableCanvasInteraction('imageCanvas');
  processingLoaderContainer.style.display = 'flex';
  if (secondDropDownChoice == 'semantic') {
    // showLoading();
  }

  // Create a Blob from the image data
  const canvas = document.createElement('canvas');
  canvas.width = imageWidth;
  canvas.height = imageHeight;
  const context = canvas.getContext('2d');
  const imgData = context.createImageData(imageWidth, imageHeight);
  imgData.data.set(imageData);
  context.putImageData(imgData, 0, 0);

  canvas.toBlob(function(blob) {
    const formData = new FormData();
    formData.append('file', blob, 'image.png');
    formData.append('imageHeight', imageHeight);
    formData.append('imageWidth', imageWidth);
    formData.append('imageProcess', selectedImageProcess);
    formData.append('imageWidthSelected', selectedImageWidth);
    formData.append('imageHeightSelected', selectedImageHeight);
    formData.append('imageTranslateDistances', translateDistances);
    formData.append('imageAffineTransform', affineTransformChoices);
    formData.append('imageRotateAngle', selectedRotateAngle);
    formData.append('imageCurrentColourSchemeMain', currentColorSchemeMain);
    formData.append('imageDesiredColorScheme', desiredColorScheme);
    formData.append('imageselectedSimpleThreshold', selectedSimpleThresholdMethod);
    formData.append('imagethresholdValue', thresholdValue);
    formData.append('imageAdaptiveParamaters', adaptiveParamaters);
    formData.append('imagethresholdMax', thresholdMax);
    formData.append('imageselectedKernel', selectedKernel);
    formData.append('imageMorphSelection', morphSelection);
    formData.append('imageContourFeatureSelection', contourFeatureSelection);
    formData.append('imageContourBoundingBoxSelection', contourBoundingBoxSelection);
    formData.append('imagefftFilterSelection', fftFilterSelection);
    formData.append('imageSelectedEdgeDetection', selectedEdgeDetection);
    formData.append('imageClusterSeg', clusterSeg);
    formData.append('imageSliderOutput', sliderChoice);
    formData.append('requestStartTime', new Date().getTime());

    // Send the image data to the server using a fetch request
    const requestStartTime = new Date().getTime();
    fetch('/process_image', {
      method: 'POST',
      body: formData
    })
    .then(response => {
      const responseTime = new Date().getTime();
      const totalTime = responseTime - requestStartTime;
      return response.json(); // Ensure the JSON is returned
    })
    .then(data => {
      const processingTime = new Date().getTime() - requestStartTime;
      
      led.classList.remove('thinking');
      led.classList.add('ready');
      processingLoaderContainer.style.display = 'none';
      imgDisplayColumn.style.position = ''; // Reset position style
      imgDisplayColumn.style.display = ''; // Reset display style
      enableCanvasInteraction('imageCanvas');
      
      // Update Current Colour Scheme
      if (secondDropDownChoice == 'semantic') {
        removeLoading();
      }
      
      tempCurrentColourScheme = data.desiredColourScheme;
      let numColumPerRow = 0;
      let nextFreeRow = getNextFreeRow();

      if (entireImage == true) {
        currentColorSchemeMain = tempCurrentColourScheme;
      }    
      
      // Deal with Histogram of the image
      if (data.histogramVals && data.histogramVals.length > 0) {
        createPlotlyHistogram(data, nextFreeRow);
        numColumPerRow += 1;
      }
      
      // Deal with fft features
      if (data.fftThresh && data.fftThresh.length > 0) {
        let nextFreeCanvasId = showNextFreeCanvas('mainCanvas', nextFreeRow);
        drawImageInCanvas(data.img, nextFreeCanvasId);
        numColumPerRow += 1;
        createFftThresh(data, nextFreeRow);
        numColumPerRow += 1;
      }
      
      if (data.semanticBool == true) {
        initializeDataTable();
      }

      if (!placeImageCanvas) {
        jsonReplaceMainImg(data);
      } else {
        // Add the image to the smaller canvases
        let nextFreeCanvasId = showNextFreeCanvas('mainCanvas', nextFreeRow);
        drawImageInCanvas(data.img, nextFreeCanvasId);
        numColumPerRow += 1;
      }

      setColumnsForRow(nextFreeRow, numColumPerRow);
    })
    .catch(error => {
      console.error('Error processing image:', error);
    });
  }, 'image/png');
}

// Function to make predictions on image
async function sendPredImage(buttonid, clickedimageProcess, fileType) {
  removeCLassPredText();
  const fileInput = document.getElementById(buttonid);


  if (fileType == 'image/jpeg' || fileType == 'image/png') {

    let predEntireImageData;
    if (buttonid != 'ocrImageUpload') {
      predEntireImageData = await getImageParams();
    } else {
      predEntireImageData = imageDataTempOCR;
    }

    showLoading();

    const canvas = document.createElement('canvas');
    canvas.width = predEntireImageData.width;
    canvas.height = predEntireImageData.height;
    const context = canvas.getContext('2d');
    const imgData = context.createImageData(predEntireImageData.width, predEntireImageData.height);
    imgData.data.set(predEntireImageData.data);
    context.putImageData(imgData, 0, 0);

    canvas.toBlob(function (blob) {
      const formData = new FormData();
      formData.append('file', blob, 'image.png');
      formData.append('predImageHeight', predEntireImageData.height);
      formData.append('predImageWidth', predEntireImageData.width);
      formData.append('imageProcess', clickedimageProcess);
      formData.append('binModel', clickedBinModel);
      formData.append('multiModel', clickedClassModel);
      formData.append('detectionModel', objDetModel);
      formData.append('selectedTask', buttonid);
      formData.append('fileType', fileType);
      formData.append('requestStartTime', new Date().getTime());

      fetch('/predict', {
        method: 'POST',
        body: formData
      })
        .then(response => response.json())
        .then(data => {
          removeLoading();
          led.classList.add('broken');
          binPred = data.binPred;
          multiPred = data.multiPred;
          showClassPreds(binPred, multiPred);

          if (buttonid == 'classImageUpload') {
            jsonReplaceMainImg(data);
            fileInput.value = '';
          }

          if (buttonid == 'segImageUpload') {
            showSegPreds(data.foundNose);
            jsonReplaceMainImg(data);
            fileInput.value = '';
          }

          if (buttonid == 'ocrImageUpload') {
            textOCRLst = data.imgTextOCR;
            ocrimgLst = data.processed_image_lst;
            updateTextOCR();
            updateImageOCR(0);
            displayPageCount(currentTextOCRIndex + 1, textOCRLst.length);
            ocrPageSelection(textOCRLst.length);
            removeLoading();
            fileInput.value = '';
          }
        })
        .catch(error => {
          console.error('Error processing image:', error);
        });
    }, 'image/png');
  } else {
    showLoading();

    fetch('/predict-pdf', {
      method: 'POST',
      body: pdfformData,
    })
      .then(response => response.json())
      .then(data => {
        textOCRLst = data.imgTxtsLst;
        ocrimgLst = data.imgsLst;
        updateTextOCR();
        updateImageOCR(0);
        displayPageCount(currentTextOCRIndex + 1, textOCRLst.length);
        ocrPageSelection(textOCRLst.length);
        removeLoading();
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }
}

// Camera ASL Functions

const canvas = document.getElementById('imageCanvas');
const ctx = canvas.getContext('2d');
const sourceImage = document.getElementById('sourceImage');
const videoElement = document.getElementById('videoElement');
const cameraToggle = document.getElementById('toggleSwitchCamera');

let streaming = false;

function removeCamera() {
  try {
    stopCamera();
    cameraToggle.textContent = "Start Camera";
    videoElement.style.display = "none"; // Hide the video element
    resetInitialImage()
    streaming = false; }
  catch {
    }
}

cameraToggle.addEventListener('click', () => {
  if (!streaming) {
    startCamera();
    cameraToggle.textContent = "Stop Camera";
    sourceImage.style.display = "none"; // Hide the image
    videoElement.style.display = "none"; // Show the video element
    streaming = true;
    displayImageInCanvas('static/user_images/Sign_alphabet_chart_abc.jpg')
  } else {
    removeCamera()
  }
});

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  videoElement.srcObject = stream;
  videoElement.play();
  
  // Draw video frames on the canvas
  requestAnimationFrame(drawVideoToCanvas);
}

function stopCamera() {
  const stream = videoElement.srcObject;
  const tracks = stream.getTracks();
  tracks.forEach(track => track.stop()); // Stop the camera stream

  videoElement.srcObject = null; // Clear the video stream
  // Reset the slider to "off"
  cameraToggle.checked = false;
}

let detectedLetter = ''; // Global variable to store the detected letter
let lastSentTime = 0; // To track the last time a frame was sent
const frameInterval = 1000; // 1 second interval between frame submissions

async function drawVideoToCanvas() {
  if (!streaming) return;

  // Adjust the canvas size to match the video feed
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  
  // Draw the current frame from the video element onto the canvas
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  // Draw a semi-transparent gray background for the text
  ctx.fillStyle = "rgba(128, 128, 128, 0.7)"; 
  ctx.fillRect(5, 5, 200, 40); 

  // Draw the detected letter on the canvas
  ctx.font = "24px Arial";
  ctx.fillStyle = "red"; // Text color
  ctx.fillText(detectedLetter, 10, 30); 

  // Capture frame at intervals
  let currentTime = performance.now();
  if (currentTime - lastSentTime >= frameInterval) {
    lastSentTime = currentTime;

    // Send the current frame to the backend for processing
    sendFrameToBackend();
  }

  // Request the next frame to keep drawing
  requestAnimationFrame(drawVideoToCanvas);
}

async function sendFrameToBackend() {
  try {
    // Capture the canvas content as an image (Base64)
    const frame = canvas.toDataURL('image/jpeg');

    // Send the frame to the backend for prediction
    const response = await fetch('/process_frame', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ frame: frame })
    });

    // Check if the response is OK
    if (!response.ok) {
      console.error('Error: Failed to fetch from backend', response.status);
      return;
    }

    // Parse the JSON response
    const data = await response.json();

    // Check if the predictions field is available in the response
    if (data && data.predictions) {
      detectedLetter = `Letter Detected: ${data.predictions}`; // Update the detected letter
      console.log(detectedLetter);
    } else {
      console.error('Error: Predictions field is missing in the response');
    }
  } catch (error) {
    console.error('Error in sending frame to backend:', error);
  }
}

// Start the video drawing loop
requestAnimationFrame(drawVideoToCanvas);

function displayImageInCanvas(imagePath) {
  const canvas = document.getElementById('mainImageCanvas1'); // Target the canvas element
  const ctx = canvas.getContext('2d'); // Get the canvas context to draw
  const img = new Image(); // Create a new Image object
  canvas.style.display = 'block';
  img.onload = function() {
    // Adjust canvas size to match image size, if necessary
    canvas.width = img.width;
    canvas.height = img.height;
    // Draw the image onto the canvas
    ctx.drawImage(img, 0, 0);
  };

  // Set the image source to trigger loading
  img.src = imagePath;
  console.log('SHOWING CANVAS')
}


// Function to disable and gray out the divs
function disableDivs() {
  document.getElementById('mainImageButtons').classList.add('disabled');
  document.getElementById('imageCanvas').classList.add('disabled');
}

// Function to enable and remove gray out from the divs
function enableDivs() {
  document.getElementById('mainImageButtons').classList.remove('disabled');
  document.getElementById('imageCanvas').classList.remove('disabled');
}
      