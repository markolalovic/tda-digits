#!/bin/bash

for f in *.png; do
    convert $f -trim "$f"
done
