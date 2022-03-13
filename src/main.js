// load the data
const digitGraphData = JSON.parse(digitGraphDataJSON);

// drawing parameters
let svg;
let chartSpacing, chartWindowWidth, chartWindowHeight;
let aspectRatio, chartWidth, chartHeight;
let digitWidth, digitHeight, digitX, digitY;
let bettiX, bettiWidth, bettiHeight;
let controlsHeight, controlsY;
const blueColor = '#0062ff';

// render on range slider input
let thresholdValue = 0;
let yInput = document.querySelector('input');
let yOutput = document.querySelector('output');

yInput.addEventListener('input', function () {
  thresholdValue = yInput.value;
  render(thresholdValue);
}, false);

init();
render();
