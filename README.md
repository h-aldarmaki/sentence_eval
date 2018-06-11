# Evaluation of Compositional Representations
Evaluation scripts and data as described in "Evaluation of Unsupervised Compositional Representations". COLING 2018. 


# Requirements #

Python 3.4 

Matlab if you want to use the avg scripts. 

# Instructions #

* unzip the data files:
```
tar -xzvf data.tar.gz
gunzip models/si_skipgram_small/siskip_300.vec.gz 
```

* Create vector representations for all datasets and store them in the approperiate `\vec` folders. You can create a script that takes two arguments (input, output) and update `cmd` in `process_files.sh` to do this automatically. An example script for creating vectors using weighted average of word embeddings is given in `sif_vectors.sh`. You will need Matlab to run it:

```
./process_files.sh
```

* Run the evaluation script. If you use `process_files.sh`, it will run the evaluation scripts after creating the word vectors. Otherwise, do the following:

```
cd scripts
python eval_all_fast.py
```

* The output will be saved in scripts/log/eval.log
