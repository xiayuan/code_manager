
#!/usr/bin/python
import numpy as np
import scipy.sparse
import pickle
import logging
from sklearn import svm
from sklearn.externals import joblib
#from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# add a line for github
# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
# dtest = xgb.DMatrix('income_predict_salary.data.binary.svm.test')
# num_round = 10

labels = []
row = []; col = []; dat = []
i = 0
for l in open('train_1w'):
    arr = l.split('\t')
    labels.append(int(arr[0]))
    content= arr[1].split()
    for it in content:
        k,v = it.split(':')
        row.append(i); col.append(int(k)); dat.append(float(v))
    i += 1
csc = scipy.sparse.csc_matrix((dat, (row,col)))

#print "starting svd...."
#svd = TruncatedSVD(n_components=200, random_state=42)
#normalizer = Normalizer(copy=False)
#lsa = make_pipeline(svd, normalizer)
#csc_new = lsa.fit_transform(csc)

print "training....."
#clf = svm.SVC()
#clf = SGDClassifier(loss="hinge", penalty="l2")
#clf = RandomForestClassifier(n_estimators=10) 
clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.5, max_depth=6, random_state=0)
X=csc                      
print X
y=np.array(labels)

clf.fit(X, y)
print 'training end'
#print 'accuracy: %s' % clf.score(X,y)
joblib.dump(clf, '/tmp/gbdt_train.pkl')


# dtrain = xgb.DMatrix(csc, label = labels)
# watchlist  = [(dtest,'eval'), (dtrain,'train')]
# bst = xgb.train(param, dtrain, num_round, watchlist)

# texts=[]
# for l in open('income_predict_salary.data.binary.svm.train'):
#     arr = l.split('\t')
#     labels.append(int(arr[0]))
#     content= arr[1].split()
#     for it in content:
#         k,v = it.split(':')
#         text.append(v)
#     texts.append(text)
# pprint(texts)
# add a line for github in the end




