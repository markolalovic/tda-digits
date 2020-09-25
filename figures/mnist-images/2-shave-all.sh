#!/bin/bash

# try it
# convert graph-0.png -shave 1x1 graph-0.png

for f in *.png; do
    convert $f -shave 1x1 "$f"
done


