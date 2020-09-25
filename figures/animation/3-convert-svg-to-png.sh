#!/bin/bash

for f in *.svg; do
    rsvg-convert $f > "$f.png"
done

