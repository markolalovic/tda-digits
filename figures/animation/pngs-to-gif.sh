#!/bin/bash

# TODO: add a few frames to make pauses
# TODO: delay 10 -> 20 -> 40
convert -delay 40 -loop 0 `ls -v *.png` anim.gif
