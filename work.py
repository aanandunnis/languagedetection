from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from nltk import word_tokenize
import nltk
import string
import random
import joblib

def basis(filedic):
  f = open(filedic)
  r = f.read()
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

bg_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/bg.txt')
cs_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/cs.txt')
da_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/da.txt')
de_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/de.txt')
el_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/el.txt')
en_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/en.txt')
es_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/es.txt')
et_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/et.txt')
fi_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/fi.txt')
fr_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/fr.txt')
hu_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/hu.txt')
it_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/it.txt')
It_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/it2.txt')
lv_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/lv.txt')
nl_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/nl.txt')
#pl_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/pl.txt')
pt_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/pt.txt')
ro_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/ro.txt')
sk_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/sk.txt')
sl_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/sl.txt')
sv_basis = basis('/Users/athuls/Desktop/unni/project/singlefile/sv.txt')

clean_bg = clean_file(bg_basis,'bg')
clean_cs = clean_file(cs_basis,'cs')
clean_da = clean_file(da_basis,'da')
clean_de = clean_file(de_basis,'de')
clean_el = clean_file(el_basis,'el')
clean_en = clean_file(en_basis,'en')
clean_es = clean_file(es_basis,'es')
clean_et = clean_file(et_basis,'et')
clean_fi = clean_file(fi_basis,'fi')
clean_fr = clean_file(fr_basis,'fr')
clean_hu = clean_file(hu_basis,'hu')
clean_it = clean_file(it_basis,'it')
clean_It = clean_file(It_basis,'It')
clean_lv = clean_file(lv_basis,'lv')
clean_nl = clean_file(nl_basis,'nl')
#clean_pl = clean_file(pl_basis,'pl')
clean_pt = clean_file(pt_basis,'pt')
clean_ro = clean_file(ro_basis,'ro')
clean_sk = clean_file(sk_basis,'sk')
clean_sl = clean_file(sl_basis,'sl')
clean_sv = clean_file(sv_basis,'sv')

train_set = clean_bg + clean_cs + clean_da + clean_de + clean_el + clean_en + clean_es + clean_et + clean_fi + clean_fr + clean_hu + clean_it + clean_It + clean_lv + clean_nl + clean_pt + clean_ro + clean_sk + clean_sl + clean_sv



classif = ModelClassifier(train_set)

filename = 'finalized_model.sav'
joblib.dump(classif, filename)







