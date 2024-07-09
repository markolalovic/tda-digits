// load the data
// const digitGraphData = JSON.parse(digitGraphDataJSON);
const digitGraphData = {
  "graph": {"name": "digit-graph", "type": "filtration", 
  "nodes": [{"id": "1", "time": 2, "metadata": {"x": 3, "y": 2}}, {"id": "2", "time": 2, "metadata": {"x": 4, "y": 2}}, {"id": "3", "time": 3, "metadata": {"x": 2, "y": 3}}, {"id": "4", "time": 3, "metadata": {"x": 5, "y": 3}}, {"id": "5", "time": 3, "metadata": {"x": 6, "y": 3}}, {"id": "6", "time": 4, "metadata": {"x": 2, "y": 4}}, {"id": "7", "time": 4, "metadata": {"x": 7, "y": 4}}, {"id": "8", "time": 5, "metadata": {"x": 2, "y": 5}}, {"id": "9", "time": 5, "metadata": {"x": 7, "y": 5}}, {"id": "10", "time": 6, "metadata": {"x": 3, "y": 6}}, {"id": "11", "time": 6, "metadata": {"x": 8, "y": 6}}, {"id": "12", "time": 7, "metadata": {"x": 4, "y": 7}}, {"id": "13", "time": 7, "metadata": {"x": 8, "y": 7}}, {"id": "14", "time": 8, "metadata": {"x": 5, "y": 8}}, {"id": "15", "time": 8, "metadata": {"x": 8, "y": 8}}, {"id": "16", "time": 9, "metadata": {"x": 6, "y": 9}}, {"id": "17", "time": 9, "metadata": {"x": 7, "y": 9}}, {"id": "18", "time": 10, "metadata": {"x": 7, "y": 10}}, {"id": "19", "time": 11, "metadata": {"x": 7, "y": 11}}, {"id": "20", "time": 12, "metadata": {"x": 7, "y": 12}}, {"id": "21", "time": 12, "metadata": {"x": 8, "y": 12}}, {"id": "22", "time": 12, "metadata": {"x": 9, "y": 12}}, {"id": "23", "time": 13, "metadata": {"x": 6, "y": 13}}, {"id": "24", "time": 13, "metadata": {"x": 10, "y": 13}}, {"id": "25", "time": 13, "metadata": {"x": 11, "y": 13}}, {"id": "26", "time": 14, "metadata": {"x": 6, "y": 14}}, {"id": "27", "time": 14, "metadata": {"x": 12, "y": 14}}, {"id": "28", "time": 15, "metadata": {"x": 6, "y": 15}}, {"id": "29", "time": 15, "metadata": {"x": 12, "y": 15}}, {"id": "30", "time": 16, "metadata": {"x": 6, "y": 16}}, {"id": "31", "time": 16, "metadata": {"x": 13, "y": 16}}, {"id": "32", "time": 17, "metadata": {"x": 7, "y": 17}}, {"id": "33", "time": 17, "metadata": {"x": 13, "y": 17}}, {"id": "34", "time": 18, "metadata": {"x": 8, "y": 18}}, {"id": "35", "time": 18, "metadata": {"x": 9, "y": 18}}, {"id": "36", "time": 18, "metadata": {"x": 10, "y": 18}}, {"id": "37", "time": 18, "metadata": {"x": 11, "y": 18}}, {"id": "38", "time": 18, "metadata": {"x": 12, "y": 18}}, {"id": "39", "time": 18, "metadata": {"x": 14, "y": 18}}, {"id": "40", "time": 19, "metadata": {"x": 14, "y": 19}}, {"id": "41", "time": 20, "metadata": {"x": 15, "y": 20}}], 
  "edges": [{"source": "1", "target": "2", "time": 2}, {"source": "1", "target": "3", "time": 3}, {"source": "2", "target": "4", "time": 3}, {"source": "4", "target": "5", "time": 3}, {"source": "3", "target": "6", "time": 4}, {"source": "5", "target": "7", "time": 4}, {"source": "6", "target": "8", "time": 5}, {"source": "7", "target": "9", "time": 5}, {"source": "8", "target": "10", "time": 6}, {"source": "9", "target": "11", "time": 6}, {"source": "10", "target": "12", "time": 7}, {"source": "11", "target": "13", "time": 7}, {"source": "12", "target": "14", "time": 8}, {"source": "13", "target": "15", "time": 8}, {"source": "14", "target": "16", "time": 9}, {"source": "15", "target": "17", "time": 9}, {"source": "16", "target": "18", "time": 10}, {"source": "17", "target": "18", "time": 10}, {"source": "18", "target": "19", "time": 11}, {"source": "19", "target": "21", "time": 12}, {"source": "20", "target": "21", "time": 12}, {"source": "21", "target": "22", "time": 12}, {"source": "20", "target": "23", "time": 13}, {"source": "22", "target": "24", "time": 13}, {"source": "24", "target": "25", "time": 13}, {"source": "23", "target": "26", "time": 14}, {"source": "25", "target": "27", "time": 14}, {"source": "26", "target": "28", "time": 15}, {"source": "27", "target": "29", "time": 15}, {"source": "28", "target": "30", "time": 16}, {"source": "29", "target": "31", "time": 16}, {"source": "30", "target": "32", "time": 17}, {"source": "31", "target": "33", "time": 17}, {"source": "32", "target": "34", "time": 18}, {"source": "34", "target": "35", "time": 18}, {"source": "35", "target": "36", "time": 18}, {"source": "36", "target": "37", "time": 18}, {"source": "37", "target": "38", "time": 18}, {"source": "33", "target": "38", "time": 18}, {"source": "33", "target": "39", "time": 18}, {"source": "39", "target": "40", "time": 19}, {"source": "40", "target": "41", "time": 20}]}
};

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
