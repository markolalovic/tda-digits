function bettiBarcodes(selection, props) {
  const {
    posX,
    posY,
    width,
    height,
    betti
  } = props;

  const xScale = d3.scaleLinear()
    .domain([0, 22])
    .range([0, width]);

  const xAxis = d3.axisBottom().scale(xScale);
  const axis = selection
    .selectAll('.axis3-' + String(betti))
    .data([null]);
  axisEnter = axis
    .enter().append('g')
      .attr('class', 'axis3-' + String(betti));
  axisEnter
    .merge(axis)
      .call(xAxis)
      .attr('transform',
        `translate(${posX}, ${posY + height - chartSpacing})`);
  axis.exit().remove();

  // draw vertical grid lines
  const gridLines = selection
    .selectAll('.vertical-grid-lines-' + String(betti))
    .data(Array.from(Array(10).keys()));
  const gridLinesEnter = gridLines
    .enter().append('line')
      .attr('class', 'vertical-grid-lines-' + String(betti))
      .attr('stroke-width', 1);
  gridLinesEnter
    .merge(gridLines)
      .attr('stroke', 'lightgray')
      .attr('x1', d => xScale((d + 1)*2))
      .attr('y1', `${posY}`)
      .attr('x2', d => xScale((d + 1)*2))
      .attr('y2', `${posY + height - chartSpacing}`)
      .attr('transform', () =>
        `translate(${posX}, ${0})`);
  gridLines.exit().remove();

  const yValue = 22;
  const birthTimes = {
      0: [2],
      1: [18, 10]
  };
  const barcode = selection
    .selectAll('.barcode-' + String(betti))
    .data(birthTimes[betti]);
  const barcodeEnter = barcode
    .enter().append('line')
      .attr('class', 'barcode-' + String(betti))
      .attr('stroke', blueColor)
      .attr('stroke-width', 10);
  barcodeEnter
    .merge(barcode)
      .attr('x1', d => xScale(d))
      .attr('y1', 0)
      .attr('x2', d => xScale(Math.max(d, thresholdValue)))
      .attr('y2', 0)
      .attr('transform', (d, i) => {
        return `translate(
          ${posX},
          ${posY
            + (1 - betti) * (0.5 * height - chartSpacing/2)
            + betti * (1 - i) * 0.2 * height + i * 0.5 * height})`;
      });
  barcode.exit().remove();

  // sweep line for betti barcodes
  const sweepLineBetti = selection
    .selectAll('.sweep-line-betti-' + String(betti))
    .data([thresholdValue]);
  const sweepLineBettiEnter = sweepLineBetti
    .enter().append('line')
      .attr('class', 'sweep-line-betti-' + String(betti))
      .attr('stroke', 'red')
      .attr('stroke-width', 2)
      .attr('x1', d => xScale(d))
      .attr('y1', `${posY}`)
      .attr('x2', d => xScale(d))
      .attr('y2', `${posY + height - chartSpacing}`)
      .attr('transform', `translate(${posX}, ${0})`);
  sweepLineBetti
    .merge(sweepLineBetti)
      .attr('x1', d => xScale(d))
      .attr('y1', `${posY}`)
      .attr('x2', d => xScale(d))
      .attr('y2', `${posY + height - chartSpacing}`)
      .attr('transform', `translate(${posX}, ${0})`);
  sweepLineBetti.exit().remove();
}
