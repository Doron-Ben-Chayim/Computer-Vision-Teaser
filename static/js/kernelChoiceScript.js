let tableDimension = 3;
let currentkernelChoice;
let selectedKernel;

const identity3 = [
  [0, 0, 0],
  [0, 1, 0],
  [0, 0, 0]
];

const identity5 = [
  [1, 0, 0, 0, 0],
  [0, 1, 0, 0, 0],
  [0, 0, 1, 0, 0],
  [0, 0, 0, 1, 0],
  [0, 0, 0, 0, 1]
];

const identity7 = [
  [1, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 1],
];

const boxblur3 = [
  [1/9, 1/9, 1/9],
  [1/9, 1/9, 1/9],
  [1/9, 1/9, 1/9]
];

const boxblur5 = [
  [1/25, 1/25, 1/25, 1/25, 1/25],
  [1/25, 1/25, 1/25, 1/25, 1/25],
  [1/25, 1/25, 1/25, 1/25, 1/25],
  [1/25, 1/25, 1/25, 1/25, 1/25],
  [1/25, 1/25, 1/25, 1/25, 1/25]
];

const boxblur7 = [
  [1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49],
  [1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49],
  [1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49],
  [1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49],
  [1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49],
  [1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49],
  [1/49, 1/49, 1/49, 1/49, 1/49,1/49,1/49]
];

const gauss3 = [
  [1/16 , 2/16, 1/16 ],
  [2/16, 4/16, 2/16],
  [1/16 , 2/16, 1/16 ]
];

const gauss5 = [
  [1/256, 4/256,  6/256,  4/256,  1/256],
  [4/256, 16/256, 24/256, 16/256, 4/256],
  [6/256, 24/256, 36/256, 24/256, 6/256],
  [4/256, 16/256, 24/256, 16/256, 4/256],
  [1/256, 4/256,  6/256,  4/256,  1/256]
];

const gauss7 = [
  [0.01499249, 0.01740063, 0.0190274 , 0.01960277, 0.0190274 , 0.01740063, 0.01499249],
  [0.01740063, 0.02019558, 0.02208365, 0.02275144, 0.02208365, 0.02019558, 0.01740063],
  [0.0190274 , 0.02208365, 0.02414823, 0.02487845, 0.02414823, 0.02208365, 0.0190274 ],
  [0.01960277, 0.02275144, 0.02487845, 0.02563076, 0.02487845, 0.02275144, 0.01960277],
  [0.0190274 , 0.02208365, 0.02414823, 0.02487845, 0.02414823, 0.02208365, 0.0190274 ],
  [0.01740063, 0.02019558, 0.02208365, 0.02275144, 0.02208365, 0.02019558, 0.01740063],
  [0.01499249, 0.01740063, 0.0190274 , 0.01960277, 0.0190274 , 0.01740063, 0.01499249]
 
];

const sobelx3 = [
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]
];

const sobelx5 = [
[ 2,  1,  0, -1, -2],
[ 2,  1,  0, -1, -2],
[ 4,  2,  0, -2, -4],
[ 2,  1,  0, -1, -2],
[ 2,  1,  0, -1, -2]
];

const sobelx7 = [
[ 3,  2,  1,  0, -1, -2, -3],
[ 3,  2,  1,  0, -1, -2, -3],
[ 3,  2,  1,  0, -1, -2, -3],
[ 6,  4,  2,  0, -2, -4, -6],
[ 3,  2,  1,  0, -1, -2, -3],
[ 3,  2,  1,  0, -1, -2, -3],
[ 3,  2,  1,  0, -1, -2, -3]
]

const sobely3 = [
  [-1, -2, -1],
  [0, 0, 0],
  [1, 2, 1]
];

const sobely5 = [
  [ 2,  2,  4,  2,  2],
  [ 1,  1,  2,  1,  1],
  [ 0,  0,  0,  0,  0],
  [-1, -1, -2, -1, -1],
  [-2, -2, -4, -2, -2]
];

const sobely7 = [
  [ 3,  3,  3,  6,  3,  3,  3],
  [ 2,  2,  2,  4,  2,  2,  2],
  [ 1,  1,  1,  2,  1,  1,  1],
  [ 0,  0,  0,  0,  0,  0,  0],
  [-1, -1, -1, -2, -1, -1, -1],
  [-2, -2, -2, -4, -2, -2, -2],
  [-3, -3, -3, -6, -3, -3, -3]
];

const prewittx3 = [
  [-1, 0, 1],
  [-1, 0, 1],
  [-1, 0, 1]
];

const prewittx5 = [
  [ 1,  1,  0, -1, -1],
  [ 1,  1,  0, -1, -1],
  [ 1,  1,  0, -1, -1],
  [ 1,  1,  0, -1, -1],
  [ 1,  1,  0, -1, -1]
];

const prewittx7 = [
  [ 1,  1,  1,  0, -1, -1, -1],
  [ 1,  1,  1,  0, -1, -1, -1],
  [ 1,  1,  1,  0, -1, -1, -1],
  [ 1,  1,  1,  0, -1, -1, -1],
  [ 1,  1,  1,  0, -1, -1, -1],
  [ 1,  1,  1,  0, -1, -1, -1],
  [ 1,  1,  1,  0, -1, -1, -1]
];

const prewitty3 = [
  [-1, -1, -1],
  [0, 0, 0],
  [1, 1, 1]
];

const prewitty5 = [
  [ 1,  1,  1,  1,  1],
  [ 1,  1,  1,  1,  1],
  [ 0,  0,  0,  0,  0],
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1]
];

const prewitty7 = [
  [ 1,  1,  1,  1,  1,  1,  1],
  [ 1,  1,  1,  1,  1,  1,  1],
  [ 1,  1,  1,  1,  1,  1,  1],
  [ 0,  0,  0,  0,  0,  0,  0],
  [-1, -1, -1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1, -1, -1]
];

const basicSharp = [
  [ 0, -1,  0],
  [-1,  5, -1],
  [ 0, -1,  0]
]

const strongBasicSharp = [
  [-1, -1, -1],
  [-1,  9, -1],
  [-1, -1, -1]

]

const unsharpMask = [
  [ 1,  4,    6,  4,  1],
  [ 4, 16,   24, 16,  4],
  [ 6, 24, -476, 24,  6],
  [ 4, 16,   24, 16,  4],
  [ 1,  4,    6,  4,  1]

]

const laplaceKernel = [
  [ 0,  1, 0],
  [ 1, -4, 1],
  [ 0,  1, 0]

]

const laplaceDiag = [
  [ 1,  1, 1],
  [ 1, -8, 1],
  [ 1,  1, 1]
]

const kernelTypesDictionary = { 
  'identity3' : identity3,
  'identity5' : identity5,
  'identity7' : identity7,
  'box3': boxblur3,
  'box5': boxblur5,
  'box7': boxblur7,
  'gaussian3': gauss3,
  'gaussian5': gauss5,
  'gaussian7': gauss7,
  "sobelx3": sobelx3,
  "sobelx5": sobelx5,
  "sobelx7": sobelx7,
  "sobely3": sobely3,
  "sobely5": sobely5,
  "sobely7": sobely7,
  "prewittx3": prewittx3,
  "prewittx5": prewittx5,
  "prewittx7": prewittx7,
  "prewitty3": prewitty3,
  "prewitty5": prewitty5,
  "prewitty7": prewitty7,
  "basicSharp3" :basicSharp,
  "strongBasicSharp3" : strongBasicSharp,
  "unsharpMask5" : unsharpMask,
  "laplaceKernel3" : laplaceKernel, 
  "laplaceDiag3" : laplaceDiag
}


function showSmoothing() {
  let smoothElement = document.querySelector("#smoothingDropDown");
  smoothElement.style.display = 'flex';
}

function removeSmoothing() {
  let smoothElement = document.querySelector("#smoothingDropDown");
  smoothElement.style.display = 'none';
}

function showEdge() {
  let edgeElement = document.querySelector("#edgeDropDown");
  edgeElement.style.display = 'flex';
}

function removeEdge() {
  let edgeElement = document.querySelector("#edgeDropDown");
  edgeElement.style.display = 'none';
}

function showSharp() {
  let edgeElement = document.querySelector("#sharpDropDown");
  edgeElement.style.display = 'flex';
}

function removeSharp() {
  let edgeElement = document.querySelector("#sharpDropDown");
  edgeElement.style.display = 'none';
}


function showSecondKernelDropdown() {
  removeSmoothing();
  removeEdge();
  removeSharp();

  var firstDropdownChoice = document.getElementById("kernelCategory").value;

  if (firstDropdownChoice == "identity") {
    console.log('identity')
    identityBool = true;
    currentkernelChoice = 'identity'
    setTableData('identity')
  } else if (firstDropdownChoice == "smoothing") {
    console.log('smoothing')
    showSmoothing()
  } else if (firstDropdownChoice == "sharpening") {
    console.log('sharpening')
    showSharp()
  } else if (firstDropdownChoice == "edgeDetection") {
    showEdge()
    console.log('edge') 
  } else if (firstDropdownChoice == "morphological") {
    console.log('morphological')
  } else if (firstDropdownChoice == "frequencyDomain") {
    console.log('frequencyDomain')
  } else if (firstDropdownChoice == "custom") {
    console.log('custom')
  } 

}

function implementSelectedSmoothing () {
  var smoothingKernelChoice = document.getElementById("smoothingKernel").value;
  currentkernelChoice = smoothingKernelChoice
  setTableData(smoothingKernelChoice)
}

function implementSelectedEdge () {
  var edgeKernelChoice = document.getElementById("edgeKernel").value;
  currentkernelChoice = edgeKernelChoice
  setTableData(edgeKernelChoice)
}

function implementSelectedSharp () {
  var sharpKernelChoice = document.getElementById("sharpKernel").value;
  console.log("sharpKernelChoice",sharpKernelChoice)
  currentkernelChoice = sharpKernelChoice
  setTableData(sharpKernelChoice)
}

function adjustInputWidths() {
  const inputs = document.querySelectorAll('#inputTable input');
  inputs.forEach(input => {
    input.style.width = ((input.value.length + 1) * 8) + 'px'; // Adjust multiplier based on font size
  });
}

// Declare the setTableData function in the global scope
function setTableData(filterName) {
  filterName = filterName + tableDimension;
  console.log("filterName", filterName);

  // Get the table element by ID
  const table = document.getElementById("inputTable");
  let kernelType = kernelTypesDictionary[filterName];

  // Iterate over rows
  for (let i = 0; i < tableDimension; i++) {
    // Iterate over cells
    for (let j = 0; j < tableDimension; j++) {
      // Assuming kernelType is an array of arrays
      const value = kernelType[i][j];

      // Get the existing input element in the cell
      const input = table.rows[i].cells[j].querySelector("input");

      if (input) {
        // If an input element already exists, update its value
        input.value = value.toFixed(4);
      } else {
        // If no input element exists, create one and set its value
        const newInput = document.createElement("input");
        newInput.type = "text";
        newInput.value = value;

        // Attach the new input element to the cell
        table.rows[i].cells[j].appendChild(newInput);
      }
    }
  }
  adjustInputWidths();
}


// Declare the getTableData function in the global scope
function getTableData() {
  const data = [];
  let isValid = true;

  // Get the table element by ID
  const table = document.getElementById("inputTable");

  // Get the number of rows and columns
  const rows = table.rows.length;
  const cols = table.rows[0].cells.length;

  // Iterate over rows
  for (let i = 0; i < rows; i++) {
      const row = [];

      // Iterate over cells
      for (let j = 0; j < cols; j++) {
          const input = table.rows[i].cells[j].querySelector("input");
          const value = input.value.trim();

          // Check if the cell value is an integer or float and not empty
          if (value === '' || !/^[-+]?\d*\.?\d+$/.test(value)) {
              isValid = false;
              input.style.borderColor = 'red'; // Optional: highlight invalid input
          } else {
              input.style.borderColor = ''; // Reset border color if valid
          }

          row.push(value);
      }

      data.push(row);
  }

  if (!isValid) {
      alert("Please ensure all cells contain valid numbers (integers or floats) and no cell is empty.");
      return;
  }

  console.log(data);
  selectedKernel = data;
  sendDataToMainWindow();
}

  function createTable(dimensions) {
    // Set the number of rows and columns
    dimensions = parseInt(dimensions);
  
    // Get the table element by ID
    const table = document.getElementById("inputTable");
  
    // Clear existing rows
    while (table.rows.length > 0) {
      table.deleteRow(0);
    }
  
    // Generate rows and cells
    for (let i = 0; i < dimensions; i++) {
      const row = table.insertRow(i);
  
      for (let j = 0; j < dimensions; j++) {
        const cell = row.insertCell(j);
        const input = document.createElement("input");
        input.type = "text";
        input.style.textAlign = "center"; 
        cell.appendChild(input);
      }
    }
  }
  

document.addEventListener("DOMContentLoaded", function () {
  tableDimension = 3;  
  createTable(tableDimension);
  });

document.addEventListener("DOMContentLoaded", function () {
  // Get the form element
  const radioForm = document.querySelector('#radioForm form');

  // Add change event listener to the form
  radioForm.addEventListener("change", function(event) {
    // Check if the changed element is a radio button
    if (event.target.type === 'radio' && event.target.name === 'dimensionSelection') {
      // Get the selected value
      tableDimension = event.target.value;

      // Log or perform actions based on the selected value
      console.log("Selected Dimension: " + tableDimension);
      createTable(parseInt(tableDimension));
      setTableData(currentkernelChoice);
    }
  });
});

function sendDataToMainWindow() {
  console.log("SENDING DATA");
  // Access the opener (main window) and call the function
  if (window.opener && !window.opener.closed) {
      window.opener.receiveDataFromPopup(selectedKernel);
  }
}

