#!/bin/bash

declare -a StringArray=("torus-surface.png" "mnist-digits.png" "torus-features.png" "mnist-features.png")

for f in ${StringArray[@]}; do
   echo "trimming $f"
   convert $f -trim "$f"
done
