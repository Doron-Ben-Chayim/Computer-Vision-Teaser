let tableDimension = 3;
let currentkernelChoice;
let selectedKernel;

const identity3 = [
  [0, 0, 0],
  [0, 1, 0],
  [0, 0, 0]
];

const identity5 = [
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]
];

const identity7 = [
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
];

const boxBlur3 = [
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1]
];

const boxBlur5 = [
  [1, 1, 1,1,1],
  [1, 1, 1,1,1],
  [1, 1, 1,1,1],
  [1, 1, 1,1,1],
  [1, 1, 1,1,1]
];

const boxBlur7 = [
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1]
];

const normBoxblur3 = [
  [1/9, 1/9, 1/9],
  [1/9, 1/9, 1/9],
  [1/9, 1/9, 1/9]
];

const normBoxblur5 = [
  [1/25, 1/25, 1/25, 1/25, 1/25],
  [1/25, 1/25, 1/25, 1/25, 1/25],
  [1/25, 1/25, 1/25, 1/25, 1/25],
  [1/25, 1/25, 1/25, 1/25, 1/25],
  [1/25, 1/25, 1/25, 1/25, 1/25]
];

const normBoxblur7 = [
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

const scharrX3 = [
  [-3,  0,  3],
  [-10, 0, 10],
  [-3,  0,  3],
]

const scharrY3 = [
  [-3, -10, -3],
  [0,   0,  0],
  [ 3,  10,  3],
]

const scharrX5 = [
  [-1,  -2,  0,  2,  1],
  [-4,  -8,  0,  8,  4],
  [-6, -12,  0, 12,  6],
  [-4,  -8,  0,  8,  4],
  [-1,  -2,  0,  2,  1]
];

const scharrY5 = [
  [-1, -4, -6, -4, -1],
  [-2, -8, -12, -8, -2],
  [ 0,  0,  0,  0,  0],
  [ 2,  8, 12,  8,  2],
  [ 1,  4,  6,  4,  1]
];

const scharrX7 = [
  [-1,  -4,  -5,  0,  5,  4,  1],
  [-6, -24, -30,  0, 30, 24,  6],
  [-15, -60, -75, 0, 75, 60, 15],
  [-20, -80, -100, 0, 100, 80, 20],
  [-15, -60, -75, 0, 75, 60, 15],
  [-6, -24, -30,  0, 30, 24,  6],
  [-1,  -4,  -5,  0,  5,  4,  1]
];

const scharrY7 = [
  [-1, -6, -15, -20, -15, -6, -1],
  [-4, -24, -60, -80, -60, -24, -4],
  [-5, -30, -75, -100, -75, -30, -5],
  [ 0,   0,   0,    0,   0,   0,  0],
  [ 5,  30,  75,  100,  75,  30,  5],
  [ 4,  24,  60,   80,  60,  24,  4],
  [ 1,   6,  15,   20,  15,   6,  1]
];

const laplacian3 = [
  [ 0,  1,  0],
  [ 1, -4,  1],
  [ 0,  1,  0]
];

const laplacian5 = [
  [ 0,  0,  1,  0,  0],
  [ 0,  1,  2,  1,  0],
  [ 1,  2, -16, 2,  1],
  [ 0,  1,  2,  1,  0],
  [ 0,  0,  1,  0,  0]
];

const laplacian7 = [
  [ 0,  0,  0,  1,  0,  0,  0],
  [ 0,  0,  1,  2,  1,  0,  0],
  [ 0,  1,  2,  3,  2,  1,  0],
  [ 1,  2,  3, -24, 3,  2,  1],
  [ 0,  1,  2,  3,  2,  1,  0],
  [ 0,  0,  1,  2,  1,  0,  0],
  [ 0,  0,  0,  1,  0,  0,  0]
];

const laplaceDiagonal3 = [
  [ 1,  1,  1],
  [ 1, -8,  1],
  [ 1,  1,  1]
];

const laplaceDiagonal5 = [
  [ 1,  1,  1,  1,  1],
  [ 1,  1,  1,  1,  1],
  [ 1,  1, -24, 1,  1],
  [ 1,  1,  1,  1,  1],
  [ 1,  1,  1,  1,  1]
];

const laplaceDiagonal7 = [
  [ 1,  1,  1,  1,  1,  1,  1],
  [ 1,  1,  1,  1,  1,  1,  1],
  [ 1,  1,  1,  1,  1,  1,  1],
  [ 1,  1,  1, -48, 1,  1,  1],
  [ 1,  1,  1,  1,  1,  1,  1],
  [ 1,  1,  1,  1,  1,  1,  1],
  [ 1,  1,  1,  1,  1,  1,  1]
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

const simpleSharpening3 = [
  [ 0, -1,  0],
  [-1,  5, -1],
  [ 0, -1,  0]
];

const simpleSharpening5 = [
  [ 0,  0, -1,  0,  0],
  [ 0, -1, -1, -1,  0],
  [-1, -1, 13, -1, -1],
  [ 0, -1, -1, -1,  0],
  [ 0,  0, -1,  0,  0]
];

const simpleSharpening7 = [
  [ 0,  0,  0, -1,  0,  0,  0],
  [ 0,  0, -1, -1, -1,  0,  0],
  [ 0, -1, -1, -1, -1, -1,  0],
  [-1, -1, -1, 25, -1, -1, -1],
  [ 0, -1, -1, -1, -1, -1,  0],
  [ 0,  0, -1, -1, -1,  0,  0],
  [ 0,  0,  0, -1,  0,  0,  0]
];


const altSharpening3 = [
  [-1, -1, -1],
  [-1,  9, -1],
  [-1, -1, -1]
];

const altSharpening5 = [
  [-1, -1, -1, -1, -1],
  [-1,  2,  2,  2, -1],
  [-1,  2,  9,  2, -1],
  [-1,  2,  2,  2, -1],
  [-1, -1, -1, -1, -1]
];

const altSharpening7 = [
  [-1, -1, -1, -1, -1, -1, -1],
  [-1,  2,  2,  2,  2,  2, -1],
  [-1,  2,  3,  3,  3,  2, -1],
  [-1,  2,  3,  9,  3,  2, -1],
  [-1,  2,  3,  3,  3,  2, -1],
  [-1,  2,  2,  2,  2,  2, -1],
  [-1, -1, -1, -1, -1, -1, -1]
];

const unsharpMasking3 = [
  [-1, -2, -1],
  [-2, 13, -2],
  [-1, -2, -1]
];

const unsharpMasking5 = [
  [ 1,  4,   6,  4,  1],
  [ 4, 16,  24, 16,  4],
  [ 6, 24, -476, 24,  6],
  [ 4, 16,  24, 16,  4],
  [ 1,  4,   6,  4,  1]
];

const unsharpMasking7 = [
  [ 0,  0, -1, -1, -1,  0,  0],
  [ 0, -1, -3, -3, -3, -1,  0],
  [-1, -3,  0,  7,  0, -3, -1],
  [-1, -3,  7, 45,  7, -3, -1],
  [-1, -3,  0,  7,  0, -3, -1],
  [ 0, -1, -3, -3, -3, -1,  0],
  [ 0,  0, -1, -1, -1,  0,  0]
];

const cross3 = [
  [0, 1, 0],
  [1, 1, 1],
  [0, 1, 0]
];

const cross5 = [
  [0, 0, 1, 0, 0],
  [0, 0, 1, 0, 0],
  [1, 1, 1, 1, 1],
  [0, 0, 1, 0, 0],
  [0, 0, 1, 0, 0]
];

const cross7 = [
  [0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1],
  [0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0]
];


const disk5 = [
  [0, 0, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [1, 1, 1, 1, 1],
  [0, 1, 1, 1, 0],
  [0, 0, 1, 0, 0]
];

const disk7 = [
  [0, 0, 1, 1, 1, 0, 0],
  [0, 1, 1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [0, 1, 1, 1, 1, 1, 0],
  [0, 0, 1, 1, 1, 0, 0]
];

const elliptical5 = [
  [0, 0, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [1, 1, 1, 1, 1],
  [0, 1, 1, 1, 0],
  [0, 0, 1, 0, 0]
];

const elliptical7 = [
  [0, 0, 0, 1, 0, 0, 0],
  [0, 1, 1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [0, 1, 1, 1, 1, 1, 0],
  [0, 0, 0, 1, 0, 0, 0]
];

const diamond5 = [
  [0, 0, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [1, 1, 1, 1, 1],
  [0, 1, 1, 1, 0],
  [0, 0, 1, 0, 0]
];

const diamond7 = [
  [0, 0, 0, 1, 0, 0, 0],
  [0, 0, 1, 1, 1, 0, 0],
  [0, 1, 1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1, 1, 1],
  [0, 1, 1, 1, 1, 1, 0],
  [0, 0, 1, 1, 1, 0, 0],
  [0, 0, 0, 1, 0, 0, 0]
];







const kernelTypesDictionary = { 
  'identity3' : identity3,
  'identity5' : identity5,
  'identity7' : identity7,
  'boxNorm3': normBoxblur3,
  'boxNorm5': normBoxblur5,
  'boxNorm7': normBoxblur7,
  'box3': boxBlur3,
  'box5': boxBlur5,
  'box7': boxBlur7,
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
  "scharrx3":scharrX3,
  "scharry3":scharrY3,
  "scharrx5":scharrX5,
  "scharry5":scharrY5,
  "scharrx7":scharrX7,
  "scharry7":scharrY7,
  "simpleSharp3" :simpleSharpening3,
  "simpleSharp5" :simpleSharpening5,
  "simpleSharp7" :simpleSharpening7,
  "altSharp3" : altSharpening3,
  "altSharp5" : altSharpening5,
  "altSharp7" : altSharpening7,
  "unsharpMask3" : unsharpMasking3,
  "unsharpMask5" : unsharpMasking5,
  "unsharpMask7" : unsharpMasking7,
  "laplaceKernel3" : laplacian3,
  "laplaceKernel5" : laplacian5,
  "laplaceKernel7" : laplacian7, 
  "laplaceDiag3" : laplaceDiagonal3,
  "laplaceDiag5" : laplaceDiagonal5,
  "laplaceDiag7" : laplaceDiagonal7,
  "cross3": cross3,
  "cross5": cross5,
  "cross7": cross7,
  "disk5": disk5,
  "disk7": disk7,
  "elipse5": elliptical5,
  "elipse7": elliptical7,
  "diamond5": diamond5,
  "diamond7": diamond7
}


function showSmoothing() {
  let smoothElement = document.querySelector("#smoothingDropDown");
  smoothElement.style.display = 'flex';
}

function showMorph() {
  let smoothElement = document.querySelector("#morphDropDown");
  smoothElement.style.display = 'flex';
}

function removeMorph() {
  let edgeElement = document.querySelector("#morphDropDown");
  edgeElement.style.display = 'none';
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
  removeMorph();

  var firstDropdownChoice = document.getElementById("kernelCategory").value;

  if (firstDropdownChoice == "identity") {

    identityBool = true;
    currentkernelChoice = 'identity'

  } else if (firstDropdownChoice == "smoothing") {

    showSmoothing()
  } else if (firstDropdownChoice == "sharpening") {

    showSharp()
  } else if (firstDropdownChoice == "edgeDetection") {
    showEdge()

  } else if (firstDropdownChoice == "morphological") {
    showMorph()

  } else if (firstDropdownChoice == "frequencyDomain") {

  } else if (firstDropdownChoice == "custom") {
    setTableData('Custom')
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

function implementSelectedMorph () {
  var morphChoice = document.getElementById("morphKernel").value;
  currentkernelChoice = morphChoice
  console.log('morph',morphChoice)
  setTableData(morphChoice)
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

