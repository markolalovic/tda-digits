function init() {
  chartWindowWidth = 1024;
  chartWindowHeight = 768;
  chartSpacing = 20;

  aspectRatio = 7/20;
  chartWidth = chartWindowWidth;
  chartHeight = aspectRatio * chartWindowWidth;

  digitWidth = 0.36 * chartWidth - 3 * chartSpacing;
  digitHeight = chartHeight - 2 * chartSpacing;
  digitX = 2 * chartSpacing;
  digitY = 2 * chartSpacing;

  bettiX = 2 * digitWidth + 3 * chartSpacing
  bettiWidth = chartWidth - bettiX - chartSpacing;
  bettiHeight = 1/3 * (chartHeight - 4 * chartSpacing);

  controlsHeight = 0.2 * chartHeight - chartSpacing;
  controlsY = 2 * chartSpacing + 2 * bettiHeight + chartSpacing;

  svg = d3.select('#chart-window')
    .append('svg')
      .attr('width', chartWindowWidth)
      .attr('height', chartWindowHeight);

  svg.call(appendImages, {
    posX: digitX,
    posY: digitY,
    width: digitWidth,
    height: digitHeight
  });

  svg.call(layoutRect, {
    rectX: digitX,
    rectY: digitY,
    rectWidth: digitWidth - 2 * chartSpacing,
    rectHeight: digitHeight
  });

  svg.call(layoutRect, {
    rectX: digitWidth + 3 * chartSpacing,
    rectY: digitY,
    rectWidth: digitWidth - 2 * chartSpacing,
    rectHeight: digitHeight
  });

  svg.call(layoutRect, {
    rectX: bettiX,
    rectY: digitY,
    rectWidth: bettiWidth,
    rectHeight: bettiHeight - chartSpacing
  });

  svg.call(layoutRect, {
    rectX: bettiX,
    rectY: digitY + bettiHeight + chartSpacing,
    rectWidth: bettiWidth,
    rectHeight: bettiHeight - chartSpacing
  });
}

function render(thresholdValue) {
  svg.call(digitGraph, {
    posX: digitWidth + 3 * chartSpacing,
    posY: digitY,
    width: digitWidth,
    height: digitHeight
  });

  svg.call(bettiBarcodes, {
    posX: bettiX,
    posY: digitY,
    width: bettiWidth,
    height: bettiHeight,
    betti: 0
  });

  svg.call(bettiBarcodes, {
    posX: bettiX,
    posY: digitY + bettiHeight + chartSpacing,
    width: bettiWidth,
    height: bettiHeight,
    betti: 1
  });
}
