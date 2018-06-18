#!/bin/bash

cd scripts
mkdir avg/scratch
sed '/^\s*$/d' $1 avg/scratch/data.txt
perl avg/tokenize.pl avg/scratch/data.txt avg/scratch/data.tok
perl avg/clean.pl avg/scratch/data.tok avg/scratch/data.clean

model_dir=../models/si_skipgram_small
perl avg/index.pl 0 avg/scratch/data.clean avg/scratch/data.ind $model_dir/vocab 

perl avg/matlab_format_sif.pl  avg/scratch/data.ind $model_dir/sif

matlab -nojvm -r "path(path,'avg/'); getAvg('avg/scratch/data.ind.ml', '$model_dir/siskip_300.vec', 'avg/scratch/data.vec'); exit"


mv avg/scratch/data.vec $2

rm -r avg/scratch
cd ..

