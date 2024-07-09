function digitGraph(selection, props) {
  const {
    posX,
    posY,
    width,
    height
  } = props;

  const xScale = d3.scaleLinear()
    .domain([0, 17])
    .range([posX, posX + width - 2*chartSpacing]);

  const yScale = d3.scaleLinear()
    .domain([0, 22])
    .range([height, 0]);

  function getLineCoord(d, coords, atPosition) {
    const u = d['source'];
    const v = d['target'];

    var p1 = coords[u];
    var p2 = coords[v];

    const lineCoords = {
      'x1': xScale(p1[0]),
      'y1': yScale(p1[1]),
      'x2': xScale(p2[0]),
      'y2': yScale(p2[1])
    };

    return lineCoords[atPosition];
  }

  // data
  const graph = digitGraphData['graph'];
  const nodes = graph['nodes'];
  const edges = graph['edges'];

  // filter data based on thresholdValue
  const filteredNodes = nodes.filter(d => d.time <= thresholdValue);
  const filteredEdges = edges.filter(d => d.time <= thresholdValue);

  // extract the coordinates of nodes
  let coords = Object();
  nodes.forEach(function(node) {
    coords[node['id']] = [
      node['metadata'].x,
      node['metadata'].y
    ];
  });

  // draw horizontal grid lines
  const gridLines = selection
    .selectAll('.horizontal-grid-lines')
    .data(Array.from(Array(10).keys()));
  const gridLinesEnter = gridLines
    .enter().append('line')
      .attr('class', 'horizontal-grid-lines')
      .attr('stroke', 'lightgray')
      .attr('stroke-width', 1);
  gridLinesEnter
    .merge(gridLines)
      .attr('x1', xScale(0))
      .attr('y1', d => yScale((d + 1)*2))
      .attr('x2', xScale(17))
      .attr('y2', d => yScale((d + 1)*2))
      .attr('transform', `translate(${0}, ${digitY})`);
  gridLines.exit().remove();

  // sweep line for handwritten digit
  const sweepLine = selection
    .selectAll('.sweep-line')
    .data([thresholdValue]);
  const sweepLineEnter = sweepLine
    .enter().append('line')
      .attr('class', 'sweep-line')
      .attr('stroke', 'red')
      .attr('stroke-width', 2);
  sweepLineEnter
    .merge(sweepLine)
      .attr('x1', digitX)
      .attr('y1', d => yScale(d))
      .attr('x2', digitWidth)
      .attr('y2', d => yScale(d))
      .attr('transform', `translate(${0}, ${digitY})`);
  sweepLine.exit().remove();

  // sweep line for digit graph
  const sweepLineDG = selection
    .selectAll('.sweep-line-digit-graph')
    .data([thresholdValue]);
  const sweepLineDGEnter = sweepLineDG
    .enter().append('line')
      .attr('class', 'sweep-line-digit-graph')
      .attr('stroke', 'red')
      .attr('stroke-width', 2);
  sweepLineDGEnter
    .merge(sweepLineDG)
      .attr('x1', xScale(0))
      .attr('y1', d => yScale(d))
      .attr('x2', xScale(17))
      .attr('y2', d => yScale(d))
      .attr('transform', `translate(${0}, ${digitY})`);
  sweepLineDG.exit().remove();

  // legend
  const legendText = selection
    .selectAll('.legend-text')
    .data([thresholdValue]);
  const legendTextEnter = legendText
    .enter().append('text')
      .attr('class', 'legend-text')
      .attr('x', 778)
      .style('fill', '#010101')
      .attr('y', 300 + 5);
  legendTextEnter
    .merge(legendText)
      .text(d => `${d}`);
  legendText.exit().remove();

  // draw the edges as lines
  if (filteredEdges) {
    const edgeLines = selection
      .selectAll('.edge-line')
      .data(filteredEdges);
    const edgeLinesEnter = edgeLines
      .enter().append('line')
        .attr('class', 'edge-line')
        .attr('stroke', blueColor)
        .attr('stroke-width', 2);
    edgeLinesEnter
      .merge(edgeLines)
        .attr('x1', d => getLineCoord(d, coords, 'x1'))
        .attr('y1', d => getLineCoord(d, coords, 'y1'))
        .attr('x2', d => getLineCoord(d, coords, 'x2'))
        .attr('y2', d => getLineCoord(d, coords, 'y2'))
        .attr('transform', `translate(${0}, ${digitY})`);
    edgeLines.exit().remove();
  }

  // draw the nodes as circles
  if (filteredNodes) {
    const nodesSelection = selection
      .selectAll('.node')
      .data(filteredNodes);
    const nodesSelectionEnter = nodesSelection
      .enter().append('circle')
        .attr('class', 'node')
        .attr('stroke', 'white')
        .style('fill', 'black')
        .attr('r', 3.3)
        .style('stroke-width', 0.1);
    nodesSelectionEnter
      .merge(nodesSelection)
        .attr('cx', d => xScale(d['metadata'].x))
        .attr('cy', d => yScale(d['metadata'].y))
        .attr('transform', `translate(${0}, ${digitY})`)
        .raise(); // raise the nodes
    nodesSelection.exit().remove();
  }

  const yAxis = d3.axisLeft().scale(yScale);
  const axis = selection
    .selectAll('.axis2')
    .data([null]);
  axisEnter = axis
    .enter().append('g')
      .attr('class', 'axis2');
  axisEnter
    .merge(axis)
      .call(yAxis)
      .attr('transform', `translate(${posX}, ${posY})`);
  axis.exit().remove();
}
