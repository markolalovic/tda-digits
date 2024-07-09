function layoutRect(selection, props) {
  const {
    rectX,
    rectY,
    rectWidth,
    rectHeight
  } = props;

  selection.append('rect')
    .attr('fill', 'none')
    .style('stroke-width', 1)
    .style('stroke-opacity', 0.8)
    .attr('stroke', 'black')
    .attr('x', rectX)
    .attr('y', rectY)
    .attr('width', rectWidth)
    .attr('height', rectHeight);
}
