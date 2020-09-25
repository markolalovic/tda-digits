#!/bin/bash

# add a few frames at the end, to make a pause at the end
cp intro86.svg.png intro87.svg.png
cp intro86.svg.png intro88.svg.png
cp intro86.svg.png intro89.svg.png
cp intro86.svg.png intro90.svg.png
cp intro86.svg.png intro91.svg.png
cp intro86.svg.png intro92.svg.png
cp intro86.svg.png intro93.svg.png
cp intro86.svg.png intro94.svg.png

convert -delay 10 -loop 0 `ls -v *.svg.png` anim.gif
