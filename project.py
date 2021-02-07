from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from nltk import word_tokenize
import nltk
import string
import random
import joblib

def basis(filedic):
  f = open(filedic,'r')
  r = f.read(encoding= 'unicode_escape')
  r = r.lower()
  r_basic = r.split()
  return r_basic[1:1000]


def clean_file(basis,lang):
  table = str.maketrans('', '', string.punctuation)
  clean = [({'word': w.translate(table)}, lang) for w in basis]
  return clean

def ModelClassifier(train_set):
  classif = SklearnClassifier(BernoulliNB()).train(train_set)
  return classif

pl_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/pl.txt')

clean_pl = clean_file(pl_basis,'pl')

classif = ModelClassifier(clean_pl)
