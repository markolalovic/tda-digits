#!/usr/bin/python3
# create-svgs.py - Creates SVG mosaik from PNG images for animation.

import numpy as np

sweep_line_range = list(np.linspace(0, 22, 90))
sweep_line_range = sweep_line_range[:-3]

i = 0
for sweep_line in sweep_line_range:
    f = open('../animation/intro' + str(i) + '.svg', 'w')

    header = '<svg height="600" width="1600"\n \
      preserveaspectratio="xMidYMid meet"\n \
      xmlns="http://www.w3.org/2000/svg"\n \
      xmlns:xlink="http://www.w3.org/1999/xlink">\n\n'

    background = '  <rect width="100%" height="100%" fill="white"/>\n\n'

    original = '  <image x="-19" y="10"\n \
         width="520" height="560" xlink:href="./original-' \
         + str(sweep_line) + '.png" />\n\n'


    graph = '  <image x="460" y="10"\n \
         width="520" height="560" xlink:href="./graph-' \
         + str(sweep_line) + '.png" />\n\n'


    betti = '  <image x="920" y="12"\n \
         width="720" height="300" xlink:href="./betti' \
         + str(sweep_line) + '.png" />\n\n'

    algorithms = '  <image x="984" y="329"\n \
        width="600" height="240" xlink:href="./algorithms-matplotlib.png" />\n'

    f.write(header)
    f.write(background)
    f.write(original)
    f.write(graph)
    f.write(betti)
    f.write(algorithms)
    f.write('</svg>')

    f.close()
    i += 1

