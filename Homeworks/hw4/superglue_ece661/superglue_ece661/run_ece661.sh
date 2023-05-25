#!/bin/bash
#img0=../../HW4-Images/Figures/books_1.jpeg
#img1=../../HW4-Images/Figures/books_2.jpeg

#img0=../../HW4-Images/Figures/building_1.jpg
#img1=../../HW4-Images/Figures/building_2.jpg

#img0=../../HW4-Images/Figures/fountain_1.jpg
#img1=../../HW4-Images/Figures/fountain_2.jpg

#img0=../../HW4-Images/Figures/garden_1.jpg
#img1=../../HW4-Images/Figures/garden_2.jpg

img0=../../../hw5/HW5-Images/0.jpg
img1=../../../hw5/HW5-Images/1.jpg

outDir=../../../hw5/HW5-Images/Results

python3 superglue_ece661.py $img0 $img1 $outDir
