#!/bin/bash

img0=../HW5-Images/0.jpg
img1=../HW5-Images/1.jpg

outDir=../HW5-Images/Results

python3 superglue_ece661.py $img0 $img1 $outDir
