# Evaluation.py
import numpy as np
import scipy as sp
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model, preprocessing
from sklearn.metrics import classification_report, accuracy_score, f1_score
import sys
import logging
import math
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD

sif=0 # enable removal of first principal component
npc=1
def compute_pc(X):
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X):
    if sif == 0:
       return X, 0
    pc = compute_pc(X)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX, pc


def remove_given_pc(X, pc):
    if sif == 0:
       return X
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX



def print_results(scores, scoring):
  for metric in scoring:
     logging.info("%s: %0.3f (+/- %0.3f)" ,
        metric, scores['test_'+metric].mean(), scores['test_'+metric].std() * 2)

def gridSearch_cv(train_x, train_y, test_x, test_y, param_grid, folds, scoring, refit, shuf, dev_ratio=0):
  cv_folds= KFold(n_splits=folds, shuffle=shuf)
  if dev_ratio > 0:
     pre_fold=np.ones(train_x.shape[0])*-1
     inds = np.random.choice(pre_fold.size, size=math.floor(dev_ratio*train_x.shape[0])) 
     pre_fold[inds]=0
     cv_folds=PredefinedSplit(test_fold=pre_fold)
  svm=SVC()
  clf = GridSearchCV(estimator=svm, param_grid=param_grid,
            cv=cv_folds, scoring=scoring, refit=refit)
  clf.fit(train_x, train_y)
  logging.info("Best parameters in grid search:")
  logging.info("")
  logging.info(clf.best_params_)
  logging.info("Test scores:")
  score=clf.score(test_x, test_y)
  logging.info("test "+refit+" score: %0.3f " , score)
  pred_y=clf.predict(test_x)
  logging.info(classification_report(test_y, pred_y))


def train_test(train_x, train_y, test_x, test_y):
  train_x, pc=remove_pc(train_x)
  test_x=remove_given_pc(test_x, pc)
  clf=SVC(kernel='linear')
  clf.fit(train_x, train_y)
  pred_y=clf.predict(test_x)
  score=accuracy_score(pred_y, test_y)
  f1=f1_score(pred_y, test_y)
  logging.info("Test set accuracy: %0.3f, f1: %0.3f" ,score, f1)

def nested_cv(data, target, param_grid, scoring, refit, folds, shuf):
  inner_folds = KFold(n_splits=folds, shuffle=shuf)
  svm=SVC()
  clf = GridSearchCV(estimator=svm, param_grid=param_grid, 
            cv=inner_folds, scoring=scoring, refit=refit)
  return cv(clf, data, target, folds, shuf,  scoring, refit) 

def cv(clf, data, target, folds, shuf,  scoring, refit):
  data, pc=remove_pc(data)
  outer_folds = KFold(n_splits=folds, shuffle=shuf)
  scores = cross_validate(clf, data, target, cv=outer_folds, scoring=scoring, 
                 return_train_score=False, n_jobs=8, pre_dispatch=16)
  print_results(scores, scoring)
  return scores['test_'+refit].mean() 

def cv_nb(data, target, folds, shuf):
  cv_folds = KFold(n_splits=folds, shuffle=shuf)
  scores=np.zeros(folds)
  f1s=np.zeros(folds)
  i=0
  for train_index, test_index in cv_folds.split(data):
     X_train, X_test = data[train_index], data[test_index]
     y_train, y_test = target[train_index], target[test_index]
     scores[i], f1s[i]=nb(X_train, y_train, X_test, y_test)
     i=i+1
  logging.info("NB CV accuracy: %0.3f, f1: %0.3f" ,scores.mean(), f1s.mean())

def nb(train_x, train_y, test_x, test_y):
  gnb = BernoulliNB() 
  gnb.fit(train_x, train_y)
  pred_y = gnb.predict(test_x)
  score=accuracy_score(pred_y, test_y)
  f1=f1_score(pred_y, test_y)
  #logging.info("Test set accuracy: %0.3f, f1: %0.3f" ,score, f1)
  return score, f1


def cv_mlp(data, target, folds, shuf):
  cv_folds = KFold(n_splits=folds, shuffle=shuf)
  scores=np.zeros(folds)
  f1s=np.zeros(folds)
  i=0
  for train_index, test_index in cv_folds.split(data):
     X_train, X_test = data[train_index], data[test_index]
     y_train, y_test = target[train_index], target[test_index]
     scores[i], f1s[i]=mlp(X_train, y_train, X_test, y_test)
     i=i+1
  logging.info("MLP CV accuracy: %0.3f, f1: %0.3f" ,scores.mean(), f1s.mean())

def mlp(train_x, train_y, test_x, test_y):
  mlp = MLPClassifier()
  mlp.fit(train_x, train_y)
  pred_y = mlp.predict(test_x)
  score=accuracy_score(pred_y, test_y)
  f1=f1_score(pred_y, test_y)
  #logging.info("Test set accuracy: %0.3f, f1: %0.3f" ,score, f1)
  return score, f1


def nested_cv_verbose(X,y, param_grid, scoring, refit, folds, shuf, rand_state):
  outer_folds = KFold(n_splits=folds, shuffle=shuf, random_state=rand_state)
  svm=SVC()
  scores=np.zeros(folds)
  i=1
  for train_index, test_index in outer_folds.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    inner_folds = KFold(n_splits=folds, shuffle=shuf, random_state=rand_state)
    clf = GridSearchCV(estimator=svm, param_grid=param_grid, cv=inner_folds, 
               scoring=scoring, refit=refit)
    clf.fit(X_train, y_train)
    logging.info("Best parameters in grid search (outer fold # %d):" ,i)
    logging.info(clf.best_params_) 
    logging.info("Grid scores on development set:")
    means = clf.cv_results_['mean_test_'+refit]
    stds = clf.cv_results_['std_test_'+refit]
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
      logging.info("%0.3f (+/-%0.03f) for %r"
              , mean, std * 2, params)
    logging.info("")
    scores[i]=clf.score(X_test, y_test) 
    logging.info("Test scores:")
    logging.info("test "+refit+" score: %0.3f " , scores[i]) 
    i=i+1
  logging.info("")
  logging.info("Final scores:")
  logging.info("CV "+ refit+" mean: %0.3f " , scores.mean())
  return scores.mean()
    
def repeated_nested_cv(data, target, param_grid, folds, scoring, refit, shuf, reps):
  scores=np.zeros(reps)
  for i in range(reps):
    logging.info("TRIAL # %d:" , i+1)
    scores[i]=nested_cv(data, target, param_grid, scoring, refit, folds, shuf)
  logging.info("Overall Mean: %0.2f " , scores.mean())
  
def multi_class_cv(train_x, train_y, test_x, test_y, folds):
  logging.info("Multinomial Logistic Regression Cross validation ... ");
  train_x,pc=remove_pc(train_x)
  test_x=remove_given_pc(test_x,pc)
  #changed this for speed
    
  #clf=LogisticRegressionCV(Cs=10, penalty='l2', cv=folds, fit_intercept=True, 
   #            solver='newton-cg', refit=True, multi_class='multinomial') 
  clf=LogisticRegression(fit_intercept=True, solver='newton-cg',multi_class='multinomial')
  le = preprocessing.LabelEncoder()
  le.fit(train_y)
  y=le.transform(train_y)
  clf.fit(train_x, y);
  pred_y=clf.predict(test_x);
  test_y=le.transform(test_y)
  score=clf.score(test_x, test_y);
  logging.info("Test set accuracy: %0.3f" , score)
  logging.info(classification_report(test_y, pred_y, labels=range(0,le.classes_.shape[0])  ,target_names=le.classes_))

def nested_multi_class(X, y, folds, labels, target_names):
  X,pc=remove_pc(X)
  logging.info("Nested Multinomial Logistic Regression Cross validation")
  inner_folds = StratifiedKFold(n_splits=folds, shuffle=True)
  #clf=LogisticRegressionCV(Cs=10, penalty='l2', cv=folds, fit_intercept=True,
   #           solver='newton-cg', refit=True, multi_class='multinomial')
  clf=LogisticRegression(fit_intercept=True, solver='newton-cg',multi_class='multinomial')
  outer_folds = StratifiedKFold(n_splits=folds, shuffle=True)
  scoring=['accuracy','f1_weighted']
  scores = cross_validate(clf, X, y, cv=outer_folds, scoring=scoring,
                 return_train_score=False, n_jobs=8, pre_dispatch=16)
  print_results(scores, scoring)


def cosine_similarity(data, target):
  data,pc=remove_pc(data)
  data_norm=preprocessing.normalize(data,norm='l2')
  a = data_norm[::2]
  b = data_norm[1::2]
  sim = np.sum(a*b, axis=1)
  pr=sp.stats.pearsonr(sim, target)
  sr=sp.stats.spearmanr(sim, target) 
  logging.info ("Pearson correlation : %0.3f  Spearman correlation: %0.3f " ,pr[0], sr[0])

 
def cosine_similarity_threshold(X_1, X_2, y, test_x, test_y):
  cv_folds = KFold(n_splits=10, shuffle=True)
  scores=np.zeros(10)
  i=0
  max_acc=0
  best_th=0
  for train_index, test_index in cv_folds.split(y):
    X_train_1, X_test_1 = X_1[train_index], X_1[test_index]
    X_train_2, X_test_2 = X_2[train_index], X_2[test_index]
    y_train, y_test = y[train_index], y[test_index]
    a = preprocessing.normalize(X_train_1, norm='l2')
    b = preprocessing.normalize(X_train_2, norm='l2')
    sim = np.sum(a*b, axis=1)
    #find minimum similarity score  of paraphrases
    p_idx=np.flatnonzero(y_train)
    min_sim=np.amin(sim[p_idx])
    #find maximum similarity score for non-paraphrases
    np_idx=np.flatnonzero(y_train==0)
    max_sim=np.amax(sim[np_idx])
    #set threshold to the average of min and max
    th=(min_sim+max_sim)/2
    a= preprocessing.normalize(X_test_1, norm='l2')
    b= preprocessing.normalize(X_test_2, norm='l2')
    sim = np.sum(a*b, axis=1)
    res=1*(sim>=th)
    acc = accuracy_score(res, y_test) 
    f1= f1_score(res, y_test)
    logging.info("fold %d threshold = %0.3f" ,i, th)
    logging.info("Accuracy : %0.3f  F1: %0.3f " ,acc, f1)
    i=i+1
    if acc > max_acc:
      max_acc=acc
      best_th=th
  data=preprocessing.normalize(test_x, norm='l2')  
  a = data[::2]
  b = data[1::2]
  sim = np.sum(a*b, axis=1)
  res=1*(sim>=best_th)
  acc = accuracy_score(res, test_y)
  f1=f1_score(res, test_y) 
  logging.info("Test results.  Threshold = %0.3f" , best_th)
  logging.info("Accuracy : %0.3f  F1: %0.3f " ,acc, f1)
 


def linear_regression(train_x, train_y, test_x, test_y):
  train_x,pc=remove_pc(train_x)
  test_x=remove_given_pc(test_x, pc)
  reg = linear_model.RidgeCV(alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
  reg.fit(train_x, train_y)
  sim=reg.predict(test_x)
  pr=sp.stats.pearsonr(sim, test_y)
  sr=sp.stats.spearmanr(sim, test_y)
  logging.info("Pearson correlation : %0.3f  Spearman correlation: %0.3f " ,pr[0], sr[0])


def logistic_regression(train_x, train_y, test_x, test_y, folds):
  train_x, pc=remove_pc(train_x)
  test_x=remove_given_pc(test_x, pc)
  #clf=LogisticRegressionCV(Cs=10, penalty='l2', dual=False, cv=folds, 
           #fit_intercept=True, solver='liblinear', refit=True)
  clf=LogisticRegression(fit_intercept=True, solver='newton-cg',multi_class='multinomial')
  X=sim_features(train_x)
  clf.fit(X, train_y)
  test_x=sim_features(test_x)
  pred_y=clf.predict(test_x)
  score=clf.score(test_x, test_y);
  f1=f1_score(pred_y, test_y)
  logging.info("Test set accuracy: %0.3f, f1: %0.3f" ,score, f1)

def sim_features(x):
  data_norm=preprocessing.normalize(x,norm='l2')
  a = data_norm[::2]
  b = data_norm[1::2]
  mult=a*b
  diff=np.abs(a-b)
  return np.concatenate((mult, diff), axis=1)
