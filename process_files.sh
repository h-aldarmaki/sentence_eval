#!/bin/sh


#path to executable bash script for converting text to vectors
cmd=./sif_vectors.sh  
rm */vec/*
dir="$(pwd)"

for file in $dir/*/txt/*.txt; do
 parentdir="$(dirname $file)"
 subdir="$(dirname $parentdir)"
 tmp=${file##*/}
 base=${tmp%.*}
 echo $cmd $file $subdir/vec/${base}.vec
 $cmd $file $subdir/vec/${base}.vec
done

cd scripts
python eval_all_fast.py
