let tableDimension = 3;
let currentkernelChoice;
let selectedKernel;

const identity3 = [
  [0, 0, 0],
  [0, 1, 0],
  [0, 0, 0]
];

const identity4 = [
  [1, 0, 0,0],
  [0, 1, 0,0],
  [0, 0, 1,0],
  [0, 0, 0,1],
];

const identity5 = [
  [1, 0, 0, 0, 0],
  [0, 1, 0, 0, 0],
  [0, 0, 1, 0, 0],
  [0, 0, 0, 1, 0],
  [0, 0, 0, 0, 1]
];

const boxblur3 = [
  [1/9, 1/9, 1/9],
  [1/9, 1/9, 1/9],
  [1/9, 1/9, 1/9]
];

const boxblur4 = [
  [1/16, 1/16, 1/16, 1/16],
  [1/16, 1/16, 1/16, 1/16],
  [1/16, 1/16, 1/16, 1/16],
  [1/16, 1/16, 1/16, 1/16]
];

const boxblur5 = [
  [1/25, 1/25, 1/25, 1/25, 1/25],
  [1/25, 1/25, 1/25, 1/25, 1/25],
  [1/25, 1/25, 1/25, 1/25, 1/25],
  [1/25, 1/25, 1/25, 1/25, 1/25],
  [1/25, 1/25, 1/25, 1/25, 1/25]
];

const gauss3 = [
  [1/16 , 2/16, 1/16 ],
  [2/16, 4/16, 2/16],
  [1/16 , 2/16, 1/16 ]
];

const gauss4 = [
  [1/256, 4/256, 2/17,4/256],
  [4/256, 1/7, 4/11,1/7],
  [2/17, 4/11, 1/2,4/11],
  [4/256, 1/7, 4/11,1/7]
];

const gauss5 = [
  [1/256, 4/256,  6/256,  4/256,  1/256],
  [4/256, 16/256, 24/256, 16/256, 4/256],
  [6/256, 24/256, 36/256, 24/256, 6/256],
  [4/256, 16/256, 24/256, 16/256, 4/256],
  [1/256, 4/256,  6/256,  4/256,  1/256]
];

const sobelx3 = [
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]
];

const sobely3 = [
  [-1, -2, -1],
  [0, 0, 0],
  [1, 2, 1]
];

const prewittx3 = [
  [-1, 0, 1],
  [-1, 0, 1],
  [-1, 0, 1]
];

const prewitty3 = [
  [-1, -1, -1],
  [0, 0, 0],
  [1, 1, 1]
];



const kernelTypesDictionary = { 
  'identity3' : identity3,
  'identity4' : identity4,
  'identity5' : identity5,
  'box3': boxblur3,
  'box4': boxblur4,
  'box5': boxblur5,
  'gaussian3': gauss3,
  'gaussian4': gauss4,
  'gaussian5': gauss5,
  "sobelx3": sobelx3,
  "sobely3": sobely3,
  "prewittx3": prewittx3,
  "prewitty3": prewitty3

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


function showSecondKernelDropdown() {
  removeSmoothing();
  removeEdge();

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
}


// Declare the getTableData function in the global scope
function getTableData() {
    const data = [];

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
        row.push(input.value);
      }

      data.push(row);
    }
    console.log(data)
    selectedKernel = data
    sendDataToMainWindow()
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
  console.log("SENDING DATA")
  // Access the opener (main window) and call the function
  window.opener.receiveDataFromPopup(selectedKernel);
}