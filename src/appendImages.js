function appendImages(selection, props) {
  const {
    posX,
    posY,
    width,
    height
  } = props;

  const yScale = d3.scaleLinear().domain([0, 22]).range([height, 0]);
  const yAxis = d3.axisLeft().scale(yScale);

  selection.append('svg:image')
      .attr('xlink:href', './images/handwritten-digit.png')
      .attr('x', posX + chartSpacing/2)
      .attr('y', posY)
      .attr('width', width - 3.3 * chartSpacing + 20)
      .attr('height', height + 20)
      .attr('transform', `translate(${-8}, ${-10})`);

  selection.append('g')
      .attr('class', 'axis')
      .call(yAxis)
      .attr('transform', `translate(${posX}, ${posY})`);

  const dy = 8;
  selection.append('svg:image')
      .attr('xlink:href', './images/A.png')
      .attr('x', 42.997)
      .attr('y', 9.104 - dy)
      .attr('width', 26.146)
      .attr('height', 23.177);

  selection.append('svg:image')
      .attr('xlink:href', './images/B.png')
      .attr('x', 371.630)
      .attr('y', 9.286 - dy)
      .attr('width', 23.698)
      .attr('height', 22.812);

  selection.append('svg:image')
      .attr('xlink:href', './images/C.png')
      .attr('x', 682.624)
      .attr('y', 8.922 - dy)
      .attr('width', 23.333)
      .attr('height', 23.333);

  selection.append('svg:image')
      .attr('xlink:href', './images/beta_0.png')
      .attr('x', 650.846)
      .attr('y', 70.633)
      .attr('width', 18.926)
      .attr('height', 17.937);

  selection.append('svg:image')
      .attr('xlink:href', './images/beta_1.png')
      .attr('x', 650.846)
      .attr('y', 182.755)
      .attr('width', 18.395)
      .attr('height', 17.937);

  selection.append('line')
    .attr('x1', 680)
    .attr('y1', 300)
    .attr('x2', 730)
    .attr('y2', 300)
    .attr('stroke', 'red')
    .attr('stroke-width', 2);
  selection.append('svg:image')
      .attr('xlink:href', './images/y.png')
      .attr('x', 740)
      .attr('y', 300 - 5)
      .attr('width', 31.54)
      .attr('height', 13.66)
}
