#!/bin/bash
for i in *.jpg
do
	convert "$i" -resize 256x256! -gravity Center -crop 224x224+0+0	"$i"
	echo "$i"
done
