# Evaluation.py
import pandas as pd
from evaluation import repeated_nested_cv, cosine_similarity,linear_regression, sim_features, gridSearch_cv, nested_multi_class, cosine_similarity_threshold, logistic_regression, multi_class_cv, cv, train_test, cv_nb, cv_mlp
import numpy as np
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
import logging
from evaluation import remove_pc, remove_given_pc
def main():
 logging.basicConfig(filename='log/eval.log', format='%(message)s', filemode='w', level=logging.DEBUG)
 logging.info('Evaluation Results:')
 params = [
  {'C': [0.1, 0.5, 1, 5], 'kernel': ['linear']}]#,
  #{'C': [0.1, 0.5, 1, 10], 'degree':[2, 3],'kernel': ['poly']},
  #{'C': [0.1, 0.5, 1, 10], 'gamma': [0.1, 0.01,0.001, 0.0001], 'kernel': ['rbf']}
#]

 clf = svm.SVC()
 scoring=['accuracy','f1']
 refit='accuracy'
 folds=10
 trials=1
 #binary CV tasks :
 tasks = ['CR','mpqa','RT-s','subj','20-ngs', '20-ngs', '20-ngs', '20-ngs']
 positive_labels = ['pos','pos','pos','subj','atheism', 'baseball','windows', 'guns']
 negative_labels = ['neg','neg','neg','obj','religion','hockey','graphics','mideast']
 logging.info("Grid search parameters:")
 logging.info(params)
 i=0
 
 for task in tasks:
   logging.info("Evaluating %s (%s, %s):", task, positive_labels[i], negative_labels[i])
   pos = np.loadtxt("../"+task+"/vec/"+positive_labels[i]+".vec")
   neg = np.loadtxt("../"+task+"/vec/"+negative_labels[i]+".vec") 
   pos_gs=np.ones(pos.shape[0])
   neg_gs=np.zeros(neg.shape[0])
   logging.info("%d positive examples, %d negative examples", pos.shape[0], neg.shape[0])
   X=np.concatenate((pos, neg), axis=0)
   y=np.concatenate((pos_gs, neg_gs), axis=0)
   logging.info("CV with default linear SVM")
   X=np.nan_to_num(X)
   X=X.astype(np.float32)
   cv(SVC(kernel='linear'),X, y, folds, True, scoring, refit) 
   i=i+1
 
 #imdb
 logging.info("Evaluating imdb:")
 pos = pd.read_table("../imdb/vec/train-pos.vec", delim_whitespace=True, header=None, na_filter=False, dtype=np.float32)
 pos=pos.values
 neg = pd.read_table("../imdb/vec/train-neg.vec", delim_whitespace=True, header=None, na_filter=False, dtype=np.float32)
 neg=neg.values
 pos_gs=np.ones(pos.shape[0])
 neg_gs=np.zeros(neg.shape[0])
 logging.info("Train set: %d positive examples, %d negative examples", pos.shape[0], neg.shape[0])
 X=np.concatenate((pos, neg), axis=0)
 y=np.concatenate((pos_gs, neg_gs), axis=0)
 pos = pd.read_table("../imdb/vec/test-pos.vec", delim_whitespace=True, header=None, na_filter=False)
 pos=pos.values
 neg = pd.read_table("../imdb/vec/test-neg.vec", delim_whitespace=True, header=None, na_filter=False)
 neg=neg.values
 pos_gs=np.ones(pos.shape[0])
 neg_gs=np.zeros(neg.shape[0])
 logging.info("Test set: %d positive examples, %d negative examples", pos.shape[0], neg.shape[0])
 X_test=np.concatenate((pos, neg), axis=0)
 y_test=np.concatenate((pos_gs, neg_gs), axis=0)
 logging.info("Training using default linear SVM")
 train_test(X, y, X_test, y_test)
 
 #multinomial topic classification (newsgroups)
 classes=['religion','atheism','guns','mideast','baseball','hockey','windows','graphics']
 i=0
 X=[]
 y=[]
 for class_name in classes:
   temp = pd.read_table("../20-ngs/vec/"+class_name+".vec", delim_whitespace=True, header=None, na_filter=False, dtype=np.float32)
   temp=temp.values
   temp_label=np.ones(temp.shape[0])*i
   logging.info("Class %s : label %d, %d examples", class_name,i, temp.shape[0])
   X.append(temp)
   y.append(temp_label)
   i=i+1
 X=np.concatenate(X, axis=0)
 y=np.concatenate(y, axis=0)
 nested_multi_class(X, y, 10, range(1,10) , classes) 
 
 #Question classification
 train = pd.read_table("../trec/vec/train.vec", delim_whitespace=True, header=None, na_filter=False)
 train=train.values
 test=pd.read_table("../trec/vec/test.vec", delim_whitespace=True, header=None, na_filter=False)
 test=test.values
 train_labels=pd.read_table("../trec/txt/train.labels", delim_whitespace=True, header=None, na_filter=False)
 test_labels=pd.read_table("../trec/txt/test.labels", delim_whitespace=True, header=None, na_filter=False)
 train_labels=train_labels.values.flatten()
 test_labels=test_labels.values.flatten()
 multi_class_cv(train, train_labels, test, test_labels, folds)

 
 #STS

 logging.info("Evaluating STS benchmark:")
 train=pd.read_table("../stsbenchmark/vec/train.vec", delim_whitespace=True, header=None, na_filter=False)
 dev=pd.read_table("../stsbenchmark/vec/dev.vec", delim_whitespace=True, header=None, na_filter=False)
 test=pd.read_table("../stsbenchmark/vec/test.vec", delim_whitespace=True, header=None, na_filter=False)
 train_gs=pd.read_table("../stsbenchmark/txt/train.gs", delim_whitespace=True, header=None, na_filter=False)
 dev_gs=pd.read_table("../stsbenchmark/txt/dev.gs", delim_whitespace=True, header=None, na_filter=False)
 test_gs=pd.read_table("../stsbenchmark/txt/test.gs", delim_whitespace=True, header=None, na_filter=False)
 train=train.values
 dev=dev.values
 test=test.values
 train_gs=train_gs.values.flatten()
 dev_gs=dev_gs.values.flatten()
 test_gs=test_gs.values.flatten()
 logging.info("Cosine Similarity (unsupervised):") 
 test=np.nan_to_num(test)
 cosine_similarity(test, test_gs)
 logging.info("Linear regression (supervised):")
 train=np.nan_to_num(train)
 dev=np.nan_to_num(dev)
 train=np.concatenate((train,dev), axis=0)
 train_gs=np.concatenate((train_gs, dev_gs), axis=0)
 linear_regression(sim_features(train), train_gs, sim_features(test), test_gs)
 
 logging.info("Evaluating SICK relatedness:")
 train=pd.read_table("../sick/vec/train.vec", delim_whitespace=True, header=None, na_filter=False)
 test=pd.read_table("../sick/vec/test.vec", delim_whitespace=True, header=None, na_filter=False)
 train_gs=pd.read_table("../sick/txt/train.rel", delim_whitespace=True, header=None, na_filter=False)
 test_gs=pd.read_table("../sick/txt/test.rel", delim_whitespace=True, header=None, na_filter=False)
 train=train.values
 test=test.values
 train_gs=train_gs.values.flatten()
 test_gs=test_gs.values.flatten()
 logging.info("Cosine Similarity (unsupervised):")
 test=np.nan_to_num(test)
 cosine_similarity(test, test_gs)
 logging.info("Linear regression (supervised):")
 train=np.nan_to_num(train)
 linear_regression(sim_features(train), train_gs, sim_features(test), test_gs)

 
   
 #parahrase   
 logging.info("Evaluating MSPR (Microsoft paraphrase dataset)")
 train=pd.read_table("../paraphrase/vec/train.vec", delim_whitespace=True, header=None, na_filter=False)
 train=train.values
 test=pd.read_table("../paraphrase/vec/test.vec", delim_whitespace=True, header=None, na_filter=False)
 test=test.values
 train_gs=pd.read_table("../paraphrase/txt/train.gs", delim_whitespace=True, header=None, na_filter=False)
 train_gs=train_gs.values.flatten()
 test_gs=pd.read_table("../paraphrase/txt/test.gs", delim_whitespace=True, header=None, na_filter=False)
 test_gs=test_gs.values.flatten()
 train=np.nan_to_num(train)
 test=np.nan_to_num(test)
 X1=train[::2]
 X2=train[1::2]
 logging.info("Cosine similarity w. thrshold tuned from train set ")
 cosine_similarity_threshold(X1, X2, train_gs, test, test_gs) 
 logging.info("") 
 logging.info("Logistic regression using %d fold CV ", folds)
 logistic_regression(train, train_gs, test, test_gs, folds) 


if __name__ == "__main__":
    main()
